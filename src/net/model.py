import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.utils import sos_idx, eos_idx


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        else:
            pass


class SoftDotAttention(nn.Module):
    def __init__(self, dim_ctx, dim_h):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim_h, dim_ctx, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, context, h, mask=None):
        '''Propagate h through the network.
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
        weighted_ctx = torch.bmm(attn3, context)  # batch x dim
        return weighted_ctx, attn


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size,
                 n_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.frame_embed = nn.Linear(1024, self.embed_size)
        self.input_dropout = nn.Dropout(0.2)
        self.video_encoder = nn.GRU(input_size=embed_size, hidden_size=hidden_size // 2, num_layers=n_layers,
                                    dropout=dropout, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout, inplace=True)

        weight_init(self)

    def forward(self, vid, vid_hidden=None):
        batch_size = vid.size(0)

        vid_embedded = self.frame_embed(vid)
        vid_embedded = self.input_dropout(vid_embedded)
        vid_out, vid_states = self.video_encoder(vid_embedded, vid_hidden)

        vid_h = vid_states.permute(1, 0, 2).contiguous().view(
            batch_size, 2, -1).permute(1, 0, 2)

        init_h = torch.cat((vid_h, vid_h), 2)
        init_c = torch.cat((vid_h, vid_h), 2)

        return (init_h, init_c), vid_out


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size_en, vocab_size_zh,
                 n_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size_en = vocab_size_en
        self.vocab_size_zh = vocab_size_zh

        self.embed_en = nn.Embedding(vocab_size_en, embed_size)
        self.embed_zh = nn.Embedding(vocab_size_zh, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.vid_attention = SoftDotAttention(embed_size, hidden_size)

        self.decoder = nn.LSTM(embed_size * 3, hidden_size,
                               n_layers, dropout=dropout, batch_first=True)

        self.fc_en = nn.Sequential(nn.Linear(self.hidden_size, self.embed_size),
                                   nn.Tanh(),
                                   nn.Dropout(p=dropout),
                                   nn.Linear(embed_size, vocab_size_en))

        self.fc_zh = nn.Sequential(nn.Linear(self.hidden_size, self.embed_size),
                                   nn.Tanh(),
                                   nn.Dropout(p=dropout),
                                   nn.Linear(embed_size, vocab_size_zh))
        weight_init(self)

    def onestep(self, input_en, input_zh, last_hidden, vid_out):
        '''
        input: (B,)
        '''
        # Get the embedding of the current input word (last output word)
        embedded_en = self.embed_en(input_en).unsqueeze(1)  # (B,1, N)
        embedded_en = self.dropout(embedded_en)

        embedded_zh = self.embed_zh(input_zh).unsqueeze(1)  # (B,1, N)
        embedded_zh = self.dropout(embedded_zh)
        # Calculate attention weights and apply to encoder outputs
        vid_ctx, vid_attn = self.vid_attention(vid_out, last_hidden[0][0])
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded_en, embedded_zh, vid_ctx], 2)  # (mb, 1, input_size)

        output, hidden = self.decoder(rnn_input, last_hidden)
        output = output.squeeze(1)  # (B, 1, N) -> (B,N)
        output_en = self.fc_en(output)
        output_zh = self.fc_zh(output)
        return output_en, output_zh, hidden, vid_attn

    def forward(self, en, zh, init_hidden, vid_out, max_len, teacher_forcing_ratio):
        batch_size_en = en.size(0)
        batch_size_zh = zh.size(0)
        output_ens = torch.zeros(batch_size_en, max_len, self.vocab_size_en).cuda()
        output_zhs = torch.zeros(batch_size_zh, max_len, self.vocab_size_zh).cuda()
        hidden = (init_hidden[0][:self.n_layers].contiguous(), init_hidden[1][:self.n_layers].contiguous())
        output_en = en.data[:, 0]  # <sos>
        output_zh = zh.data[:, 0]  # <sos>
        for t in range(1, max_len):
            output_en, output_zh, hidden, attn_weights = self.onestep(output_en, output_zh, hidden, vid_out,
                                                                      )  # (mb, vocab) (1, mb, N) (mb, 1, seqlen)
            output_ens[:, t, :] = output_en
            output_zhs[:, t, :] = output_zh
            is_teacher = random.random() < teacher_forcing_ratio
            top1_en = output_en.data.max(1)[1]
            output_en = (en.data[:,
                         t] if is_teacher else top1_en).cuda()  # output should be indices to feed into nn.embedding at next step
            top1_zh = output_zh.data.max(1)[1]
            output_zh = (zh.data[:,
                         t] if is_teacher else top1_zh).cuda()  # output should be indices to feed into nn.embedding at next step
        return output_ens, output_zhs

    def inference(self, en, zh, init_hidden, vid_out, max_len, teacher_forcing_ratio=0):
        '''
        Greedy decoding
        '''
        batch_size_en = en.size(0)
        batch_size_zh = zh.size(0)
        output_ens = torch.zeros(batch_size_en, max_len, self.vocab_size_en).cuda()
        output_zhs = torch.zeros(batch_size_zh, max_len, self.vocab_size_zh).cuda()

        hidden = (init_hidden[0][:self.n_layers].contiguous(), init_hidden[1][:self.n_layers].contiguous())
        output_en = en.data[:, 0]  # <sos>
        output_zh = zh.data[:, 0]  # <sos>
        pred_lengths_en = [0] * batch_size_en
        pred_lengths_zh = [0] * batch_size_zh
        for t in range(1, max_len):
            output_en, output_zh, hidden, attn_weights = self.onestep(output_en, output_zh, hidden, vid_out,
                                                                      )  # (mb, vocab) (1, mb, N) (mb, 1, seqlen)
            output_ens[:, t, :] = output_en
            output_zhs[:, t, :] = output_zh
            is_teacher = random.random() < teacher_forcing_ratio
            top1_en = output_en.data.max(1)[1]
            output_en = (en.data[:, t] if is_teacher else top1_en).cuda()
            top1_zh = output_zh.data.max(1)[1]
            output_zh = (zh.data[:, t] if is_teacher else top1_zh).cuda()

            for i in range(batch_size_en):
                if output_en[i] == 3 and pred_lengths_en[i] == 0:
                    pred_lengths_en[i] = t
            for i in range(batch_size_zh):
                if output_zh[i] == 3 and pred_lengths_zh[i] == 0:
                    pred_lengths_zh[i] = t

        for i in range(batch_size_en):
            if pred_lengths_en[i] == 0:
                pred_lengths_en[i] = max_len
        for i in range(batch_size_zh):
            if pred_lengths_zh[i] == 0:
                pred_lengths_zh[i] = max_len
        return output_ens, pred_lengths_en, output_zhs, pred_lengths_zh

    def beam_decoding(self, video, init_hidden, vid_out, max_len, beam_size=5):
        batch_size = video.size(0)
        hidden = (init_hidden[0][:self.n_layers].contiguous(), init_hidden[1][:self.n_layers].contiguous())

        seq_en = torch.LongTensor(max_len, batch_size).zero_()
        seq_log_probs_en = torch.FloatTensor(max_len, batch_size)
        seq_zh = torch.LongTensor(max_len, batch_size).zero_()
        seq_log_probs_zh = torch.FloatTensor(max_len, batch_size)

        for i in range(batch_size):
            # treat the problem as having a batch size of beam_size
            vid_out_i = vid_out[i].unsqueeze(0).expand(beam_size, vid_out.size(1), vid_out.size(2)).contiguous()
            hidden_i = [_[:, i, :].unsqueeze(1).expand(_.size(0), beam_size, _.size(2)).contiguous() for _ in
                        hidden]  # (n_layers, bs, 1024)

            output_en = torch.LongTensor([sos_idx] * beam_size).cuda()
            output_zh = torch.LongTensor([sos_idx] * beam_size).cuda()

            output_en, output_zh, hidden_i, attn_weights = self.onestep(output_en, output_zh, hidden_i,
                                                                        vid_out_i)  # (mb, vocab) (1, mb, N) (mb, 1, seqlen)
            log_probs_en = F.log_softmax(output_en, dim=1)
            log_probs_en[:, -1] = log_probs_en[:, -1] - 1000
            neg_log_probs_en = -log_probs_en
            log_probs_zh = F.log_softmax(output_zh, dim=1)
            log_probs_zh[:, -1] = log_probs_zh[:, -1] - 1000
            neg_log_probs_zh = -log_probs_zh

            all_outputs_en = np.ones((1, beam_size), dtype='int32')
            all_masks_en = np.ones_like(all_outputs_en, dtype="float32")
            all_costs_en = np.zeros_like(all_outputs_en, dtype="float32")
            all_outputs_zh = np.ones((1, beam_size), dtype='int32')
            all_masks_zh = np.ones_like(all_outputs_zh, dtype="float32")
            all_costs_zh = np.zeros_like(all_outputs_zh, dtype="float32")

            for j in range(max_len):
                if all_masks_en[-1].sum() == 0:
                    break

                next_costs_en = (
                        all_costs_en[-1, :, None] + neg_log_probs_en.data.cpu().numpy() * all_masks_en[-1, :, None])
                (finished_en,) = np.where(all_masks_en[-1] == 0)
                next_costs_en[finished_en, 1:] = np.inf
                next_costs_zh = (
                        all_costs_zh[-1, :, None] + neg_log_probs_zh.data.cpu().numpy() * all_masks_zh[-1, :, None])
                (finished_zh,) = np.where(all_masks_zh[-1] == 0)
                next_costs_zh[finished_zh, 1:] = np.inf

                (indexes_en, outputs_en), chosen_costs_en = self._smallest(
                    next_costs_en, beam_size, only_first_row=j == 0)
                (indexes_zh, outputs_zh), chosen_costs_zh = self._smallest(
                    next_costs_zh, beam_size, only_first_row=j == 0)

                new_state_d_en = [_.data.cpu().numpy()[:, indexes_en, :]
                                  for _ in hidden_i]
                new_state_d_zh = [_.data.cpu().numpy()[:, indexes_zh, :]
                                  for _ in hidden_i]

                all_outputs_en = all_outputs_en[:, indexes_en]
                all_masks_en = all_masks_en[:, indexes_en]
                all_costs_en = all_costs_en[:, indexes_en]
                all_outputs_zh = all_outputs_zh[:, indexes_zh]
                all_masks_zh = all_masks_zh[:, indexes_zh]
                all_costs_zh = all_costs_zh[:, indexes_zh]

                output_en = torch.from_numpy(outputs_en).cuda()
                output_zh = torch.from_numpy(outputs_zh).cuda()
                hidden_i_en = self.from_numpy(new_state_d_en)
                hidden_i_zh = self.from_numpy(new_state_d_zh)
                hidden_i = hidden_i_zh
                output_en, output_zh, hidden_i, attn_weights = self.onestep(output_en, output_zh, hidden_i, vid_out_i)
                log_probs_en = F.log_softmax(output_en, dim=1)
                log_probs_zh = F.log_softmax(output_zh, dim=1)

                log_probs_en[:, -1] = log_probs_en[:, -1] - 1000
                neg_log_probs_en = -log_probs_en

                log_probs_zh[:, -1] = log_probs_zh[:, -1] - 1000
                neg_log_probs_zh = -log_probs_zh

                all_outputs_en = np.vstack([all_outputs_en, outputs_en[None, :]])
                all_costs_en = np.vstack([all_costs_en, chosen_costs_en[None, :]])
                mask_en = outputs_en != 0
                all_masks_en = np.vstack([all_masks_en, mask_en[None, :]])
                all_outputs_zh = np.vstack([all_outputs_zh, outputs_zh[None, :]])
                all_costs_zh = np.vstack([all_costs_zh, chosen_costs_zh[None, :]])
                mask_zh = outputs_zh != 0
                all_masks_zh = np.vstack([all_masks_zh, mask_zh[None, :]])

            all_outputs_en = all_outputs_en[1:]
            all_costs_en = all_costs_en[1:] - all_costs_en[:-1]
            all_masks_en = all_masks_en[:-1]
            costs_en = all_costs_en.sum(axis=0)
            lengths_en = all_masks_en.sum(axis=0)
            normalized_cost_en = costs_en / lengths_en
            best_idx_en = np.argmin(normalized_cost_en)

            all_outputs_zh = all_outputs_zh[1:]
            all_costs_zh = all_costs_zh[1:] - all_costs_zh[:-1]
            all_masks_zh = all_masks_zh[:-1]
            costs_zh = all_costs_zh.sum(axis=0)
            lengths_zh = all_masks_zh.sum(axis=0)
            normalized_cost_zh = costs_zh / lengths_zh
            best_idx_zh = np.argmin(normalized_cost_zh)

            seq_en[:all_outputs_en.shape[0], i] = torch.from_numpy(
                all_outputs_en[:, best_idx_en])
            seq_log_probs_en[:all_costs_en.shape[0], i] = torch.from_numpy(
                all_costs_en[:, best_idx_en])
            seq_zh[:all_outputs_zh.shape[0], i] = torch.from_numpy(
                all_outputs_zh[:, best_idx_zh])
            seq_log_probs_zh[:all_costs_zh.shape[0], i] = torch.from_numpy(
                all_costs_zh[:, best_idx_zh])

        seq_en, seq_log_probs_en = seq_en.transpose(0, 1), seq_log_probs_en.transpose(0, 1)
        seq_zh, seq_log_probs_zh = seq_zh.transpose(0, 1), seq_log_probs_zh.transpose(0, 1)

        pred_lengths_en = [0] * batch_size
        for i in range(batch_size):
            if sum(seq_en[i] == eos_idx) == 0:
                pred_lengths_en[i] = max_len
            else:
                pred_lengths_en[i] = (seq_en[i] == eos_idx).nonzero()[0][0]

        pred_lengths_zh = [0] * batch_size
        for i in range(batch_size):
            if sum(seq_zh[i] == eos_idx) == 0:
                pred_lengths_zh[i] = max_len
            else:
                pred_lengths_zh[i] = (seq_zh[i] == eos_idx).nonzero()[0][0]

        # return the samples and their log likelihoods
        return seq_en, pred_lengths_en, seq_zh, pred_lengths_zh  # seq_log_probs

    def from_numpy(self, states):
        return [torch.from_numpy(state).cuda() for state in states]

    @staticmethod
    def _smallest(matrix, k, only_first_row=False):
        if only_first_row:
            flatten = matrix[:1, :].flatten()
        else:
            flatten = matrix.flatten()
        args = np.argpartition(flatten, k)[:k]
        args = args[np.argsort(flatten[args])]
        return np.unravel_index(args, matrix.shape), flatten[args]
