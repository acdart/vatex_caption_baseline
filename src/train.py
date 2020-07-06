import argparse
import datetime
import json
import logging
import os
import time

import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from net.model import Encoder, Decoder
from utils.data import create_split_loaders
from utils.utils import set_logger, read_vocab, write_vocab, build_vocab, Tokenizer, padding_idx, clip_gradient

cc = SmoothingFunction()

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])


def save_checkpoint(state, cp_file):
    torch.save(state, cp_file)


def count_paras(encoder, decoder, logging=None):
    '''
    Count model parameters.
    '''
    nparas_enc = sum(p.numel() for p in encoder.parameters())
    nparas_dec = sum(p.numel() for p in decoder.parameters())
    nparas_sum = nparas_enc + nparas_dec
    if logging is None:
        print('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc / 1e6, nparas_dec / 1e6,
                                                                      nparas_sum / 1e6))
    else:
        logging.info('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc / 1e6, nparas_dec / 1e6,
                                                                             nparas_sum / 1e6))


def setup(args, clear=False):
    '''
    Build vocabs from train or train/val set.
    '''
    TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_ZH
    if clear:  ## delete previous vocab
        for file in [TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH]:
            if os.path.exists(file):
                os.remove(file)
    # Build English vocabs
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'), TRAIN_VOCAB_EN)
    # build Chinese vocabs
    if not os.path.exists(TRAIN_VOCAB_ZH):
        write_vocab(build_vocab(args.DATA_DIR, language='zh'), TRAIN_VOCAB_ZH)

    # set up seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def main(args):
    model_prefix = '{}_{}'.format(args.model_type, args.train_id)

    log_path = args.LOG_DIR + model_prefix + '/'
    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth.tar"
    init_epoch = 0

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ## set up the logger
    set_logger(os.path.join(log_path, 'train.log'))

    ## save argparse parameters
    with open(log_path + 'args.yaml', 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))

    logging.info('Training model: {}'.format(model_prefix))

    ## set up vocab txt
    setup(args, clear=True)
    print(args.__dict__)

    # indicate src and tgt language
    en, zh = 'en', 'zh'

    maps = {'en': args.TRAIN_VOCAB_EN, 'zh': args.TRAIN_VOCAB_ZH}
    vocab_en = read_vocab(maps[en])
    tok_en = Tokenizer(language=en, vocab=vocab_en, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_zh = read_vocab(maps[zh])
    tok_zh = Tokenizer(language=zh, vocab=vocab_zh, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size en/zh:{}/{}'.format(len(vocab_en), len(vocab_zh)))

    ## Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_en, tok_zh), args.batch_size,
                                                                 args.MAX_VID_LENGTH, (en, zh), num_workers=4,
                                                                 pin_memory=True)
    logging.info('train/val/test size: {}/{}/{}'.format(len(train_loader), len(val_loader), len(test_loader)))

    ## init model
    encoder = Encoder(embed_size=args.wordembed_dim,
                      hidden_size=args.enc_hid_size).cuda()
    decoder = Decoder(embed_size=args.wordembed_dim, hidden_size=args.dec_hid_size,
                      vocab_size_en=len(vocab_en), vocab_size_zh=len(vocab_zh)).cuda()

    encoder.train()
    decoder.train()

    ## define loss
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx).cuda()
    ## init optimizer
    dec_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                     lr=args.decoder_lr, weight_decay=args.weight_decay)
    enc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                     lr=args.encoder_lr, weight_decay=args.weight_decay)

    count_paras(encoder, decoder, logging)

    ## track loss during training
    total_train_loss, total_val_loss = [], []
    best_val_bleu_en, best_val_bleu_zh, best_epoch = 0, 0, 0

    ## init time
    zero_time = time.time()

    # Begin training procedure
    earlystop_flag = False
    rising_count = 0

    for epoch in range(init_epoch, args.epochs):
        ## train for one epoch
        start_time = time.time()
        train_loss = train(train_loader, encoder, decoder, criterion, enc_optimizer, dec_optimizer, epoch)

        val_loss, sentbleu_en, corpbleu_en = validate(val_loader, encoder, decoder, criterion, tok_en, tok_zh)
        end_time = time.time()

        epoch_time = end_time - start_time
        total_time = end_time - zero_time

        logging.info(
            'Total time used: %s Epoch %d time uesd: %s train loss: %.4f val loss: %.4f corpbleu-en: %.4f' % (
                str(datetime.timedelta(seconds=int(total_time))),
                epoch, str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss, corpbleu_en))

        if corpbleu_en > best_val_bleu_en:
            best_val_bleu_en = corpbleu_en
            save_checkpoint({'epoch': epoch,
                             'enc_state_dict': encoder.state_dict(), 'dec_state_dict': decoder.state_dict(),
                             'enc_optimizer': enc_optimizer.state_dict(), 'dec_optimizer': dec_optimizer.state_dict(),
                             }, cp_file)
            best_epoch = epoch

        logging.info("Finished {0} epochs of training".format(epoch + 1))

        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)

    logging.info(
        'Best corpus bleu score en-{:.4f} at epoch {}'.format(best_val_bleu_en, best_epoch))

    ### the best model is the last model saved in our implementation
    logging.info('************ Start eval... ************')
    eval(test_loader, encoder, decoder, cp_file, tok_en, tok_zh, result_path)


def train(train_loader, encoder, decoder, criterion, enc_optimizer, dec_optimizer, epoch):
    '''
    Performs one epoch's training.
    '''
    encoder.train()
    decoder.train()

    avg_loss = 0
    for cnt, (encap, zhcap, video, caplen_en, caplen_zh, enrefs, zhrefs) in enumerate(train_loader, 1):

        encap, zhcap, video, caplen_en, caplen_zh = encap.cuda(), zhcap.cuda(), video.cuda(), caplen_en.cuda(), caplen_zh.cuda()

        init_hidden, vid_out = encoder(
            video)  # fea: decoder input from encoder, should be of size (mb, encout_dim) = (mb, decoder_dim)
        scores_en, scores_zh = decoder(encap, zhcap, init_hidden, vid_out, args.MAX_INPUT_LENGTH,
                                       teacher_forcing_ratio=args.teacher_ratio)

        targets_en = encap[:,
                     1:]  # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        loss_en = criterion(scores_en[:, 1:].contiguous().view(-1, decoder.vocab_size_en),
                            targets_en.contiguous().view(-1))
        targets_zh = zhcap[:,
                     1:]  # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        loss_zh = criterion(scores_zh[:, 1:].contiguous().view(-1, decoder.vocab_size_zh),
                            targets_zh.contiguous().view(-1))

        # Back prop.
        dec_optimizer.zero_grad()
        if enc_optimizer is not None:
            enc_optimizer.zero_grad()
        loss = loss_en + loss_zh
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(dec_optimizer, args.grad_clip)
            clip_gradient(enc_optimizer, args.grad_clip)

        # Update weights
        dec_optimizer.step()
        enc_optimizer.step()

        # Keep track of metrics
        avg_loss += loss.item()

    return avg_loss / cnt


def validate(val_loader, encoder, decoder, criterion, tok_en, tok_zh):
    '''
    Performs one epoch's validation.
    '''
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    references_en = list()  # references (true captions) for calculating corpus BLEU-4 score
    hypotheses_en = list()  # hypotheses (predictions)

    references_zh = list()  # references (true captions) for calculating corpus BLEU-4 score
    hypotheses_zh = list()  # hypotheses (predictions)

    avg_loss = 0

    with torch.no_grad():
        # Batches
        for cnt, (encap, zhcap, video, caplen_en, caplen_zh, enrefs, zhrefs) in enumerate(val_loader, 1):
            encap, zhcap, video, caplen_en, caplen_zh = encap.cuda(), zhcap.cuda(), video.cuda(), caplen_en.cuda(), caplen_zh.cuda()

            # Forward prop.
            init_hidden, vid_out = encoder(
                video)  # fea: decoder input from encoder, should be of size (mb, encout_dim) = (mb, decoder_dim)
            scores_en, pred_lengths_en, scores_zh, pred_lengths_zh = decoder.inference(encap, zhcap, init_hidden,
                                                                                       vid_out,
                                                                                       args.MAX_INPUT_LENGTH)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets_en = encap[:, 1:]
            scores_copy_en = scores_en.clone()
            targets_zh = zhcap[:, 1:]
            scores_copy_zh = scores_zh.clone()

            # Calculate loss
            loss_en = criterion(scores_en[:, 1:].contiguous().view(-1, decoder.vocab_size_en),
                                targets_en.contiguous().view(-1))
            loss_zh = criterion(scores_zh[:, 1:].contiguous().view(-1, decoder.vocab_size_zh),
                                targets_zh.contiguous().view(-1))

            # Hypotheses
            _, preds_en = torch.max(scores_copy_en, dim=2)
            preds_en = preds_en.tolist()
            temp_preds_en = list()
            for j, p in enumerate(preds_en):
                temp_preds_en.append(preds_en[j][1:pred_lengths_en[j]])  # remove pads and idx-0

            preds_en = temp_preds_en
            hypotheses_en.extend(preds_en)  # preds= [1,2,3]

            enrefs = [list(map(int, i.split())) for i in enrefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]

            for r in enrefs:
                references_en.append([r])

            assert len(references_en) == len(hypotheses_en)

            _, preds_zh = torch.max(scores_copy_zh, dim=2)
            preds_zh = preds_zh.tolist()
            temp_preds_zh = list()
            for j, p in enumerate(preds_zh):
                temp_preds_zh.append(preds_zh[j][1:pred_lengths_zh[j]])  # remove pads and idx-0

            preds_zh = temp_preds_zh
            hypotheses_zh.extend(preds_zh)  # preds= [1,2,3]

            zhrefs = [list(map(int, i.split())) for i in zhrefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]

            for r in zhrefs:
                references_zh.append([r])

            assert len(references_zh) == len(hypotheses_zh)

            avg_loss += loss_en.item() + loss_zh.item()

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>

            # Calculate loss

            # Hypotheses

        # Calculate metrics
        avg_loss = avg_loss / cnt

        scorers = {
            "Bleu": Bleu(4),
            "Meteor": Meteor(),
            "Rouge": Rouge(),
            "Cider": Cider(),
            "Spice": Spice()
        }

        gts_en = {}
        res_en = {}
        for i in range(len(references_en)):
            gts_en[i] = [tok_en.decode_sentence(references_en[i][0])]
            res_en[i] = [tok_en.decode_sentence(hypotheses_en[i])]
        scores = {}
        for name, scorer in scorers.items():
            score, all_scores = scorer.compute_score(gts_en, res_en)
            if isinstance(score, list):
                for i, sc in enumerate(score, 1):
                    scores[name + str(i)] = sc
            else:
                scores[name] = score
        print("Score of EN:")
        print(scores)

        """
        gts_zh = {}
        res_zh = {}
        for i in range(len(references_zh)):
            gts_zh[i] = [tok_zh.decode_sentence(references_zh[i][0])]
            res_zh[i] = [tok_zh.decode_sentence(hypotheses_zh[i])]
        scores = {}
        for name, scorer in scorers.items():
            score, all_scores = scorer.compute_score(gts_zh, res_zh)
            if isinstance(score, list):
                for i, sc in enumerate(score, 1):
                    scores[name + str(i)] = sc
            else:
                scores[name] = score
        print("Score of ZH:")
        print(scores)
        """
        corpbleu_en = corpus_bleu(references_en, hypotheses_en)
        sentbleu_en = 0
        for i, (r, h) in enumerate(zip(references_en, hypotheses_en), 1):
            sentbleu_en += sentence_bleu(r, h, smoothing_function=cc.method7)
        sentbleu_en /= i

    return avg_loss, sentbleu_en, corpbleu_en


def eval(test_loader, encoder, decoder, cp_file, tok_en, tok_zh, result_path):
    '''
    Testing the model
    '''
    ### the best model is the last model saved in our implementation
    epoch = torch.load(cp_file)['epoch']
    logging.info('Use epoch {0} as the best model for testing'.format(epoch))
    encoder.load_state_dict(torch.load(cp_file)['enc_state_dict'])
    decoder.load_state_dict(torch.load(cp_file)['dec_state_dict'])

    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    ids_en = list()  # sentence ids
    hypotheses_en = list()  # hypotheses (predictions)
    ids_zh = list()  # sentence ids
    hypotheses_zh = list()  # hypotheses (predictions)
    with torch.no_grad():
        # Batches
        for cnt, (video, sent_id) in enumerate(test_loader, 1):
            video = video.cuda()

            # Forward prop.
            init_hidden, vid_out = encoder(
                video)  # fea: decoder input from encoder, should be of size (mb, encout_dim) = (mb, decoder_dim)
            preds_en, pred_lengths_en, preds_zh, pred_lengths_zh = decoder.beam_decoding(video, init_hidden, vid_out,
                                                                                         args.MAX_INPUT_LENGTH,
                                                                                         beam_size=5)

            # Hypotheses
            preds_en = preds_en.tolist()
            temp_preds_en = list()
            for j, p in enumerate(preds_en):
                temp_preds_en.append(preds_en[j][:pred_lengths_en[j]])  # remove pads and idx-0

            preds_en = [tok_en.decode_sentence(t) for t in temp_preds_en]

            hypotheses_en.extend(preds_en)  # preds= [[1,2,3], ... ]
            # Hypotheses
            preds_zh = preds_zh.tolist()
            temp_preds_zh = list()
            for j, p in enumerate(preds_zh):
                temp_preds_zh.append(preds_zh[j][:pred_lengths_zh[j]])  # remove pads and idx-0

            preds_zh = [tok_zh.decode_sentence(t) for t in temp_preds_zh]

            hypotheses_zh.extend(preds_zh)  # preds= [[1,2,3], ... ]

            ids_en.extend(sent_id)
            ids_zh.extend(sent_id)

    ## save to json for submission
    dc_en = dict(zip(ids_en, hypotheses_en))
    print(len(dc_en))
    dc_zh = dict(zip(ids_zh, hypotheses_zh))
    print(len(dc_zh))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + 'submission_en.json', 'w') as fp:
        json.dump(dc_en, fp)
    with open(result_path + 'submission_zh.json', 'w') as fp:
        json.dump(dc_zh, fp)
    return dc_en


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VATEX Captioning')
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml

        args = Arguments(yaml.load(fin))
    main(args)
