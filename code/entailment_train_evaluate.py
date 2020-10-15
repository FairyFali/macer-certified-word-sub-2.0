# -*- coding: utf-8 -*-#
# Name:         entailment_train_evaluate
# Description:  
# Author:       Fali Wang
# Date:         2020/10/9 14:55
import argparse
import logging
import os
import random
import numpy as np
import tqdm
import glob
from nltk import word_tokenize
import pickle

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW, \
    get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

import utils
from model import BOWModel, DecompAttentionModel, BertBasedModel, ESIM
from entailment import SNLIProcessor

logger = logging.getLogger(__name__)

processors = {
    "snli": SNLIProcessor
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def load_vocab(args, glove_path):
    logger.info('loading glove vocab...')
    task_class = processors[args.task_name]()
    vocab = {}
    embed_size = 300
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            data = line.split()
            word = data[0]
            try:
                vec = list(map(float, data[1:]))
            except:
                continue
            vocab[word] = {'vec':vec, 'freq':1}

    word2index = {'<pad>': 0, '<unk>': 1}
    index2word = ['<pad>', '<unk>']
    train_examples = task_class.get_train_examples(args.data_dir)
    dev_examples = task_class.get_dev_examples(args.data_dir)
    for example in train_examples + dev_examples:
        text_a = example.text_a
        text_b = example.text_b
        tokens = word_tokenize(text_a.lower()) + word_tokenize(text_b.lower())
        for token in tokens:
            if token in vocab:
                vocab[token]['freq'] += 1

    for token, v in vocab.items():
        if v['freq'] > 1:
            word2index[token] = len(index2word)
            index2word.append(token)
    print('vocab size', len(index2word))  # 32551
    word_mat = np.zeros([len(index2word), embed_size])
    for word in index2word:
        if word in vocab:
            word_mat[word2index[word]] = vocab[word]['vec']

    vocab = {'index2word': index2word, 'word2index': word2index, 'word_mat': word_mat}
    with open(args.cache_dir + '/{}_vocab_train.pkl'.format(args.task_name), 'wb') as f:
        pickle.dump(vocab, f)

    return index2word, word2index, word_mat


def load_and_cache_bert_example(args, tokenizer, type='train'):
    '''
    load or cache the InputExample, return dataset
    :param type:
    :param args:
    :param task_class:
    :param tokenizer:
    :return:
    '''
    task_class = processors[args.task_name]()
    # file: cached_train_bert-base-uncased_256_imdb
    file = 'normal_bert_cached_{}_{}_{}'.format(
        type,
        args.max_seq_length,
        args.task_name
    )
    cached_features_file = os.path.join(args.cache_dir, file)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = task_class.get_labels()

        if type == 'train':
            examples = task_class.get_train_examples(args.data_dir)
        elif type == 'dev':
            examples = task_class.get_dev_examples(args.data_dir)
        else:
            examples = task_class.get_test_examples(args.data_dir)
        features = utils.convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
                                                      cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                      # xlnet has a cls token at the end
                                                      cls_token=tokenizer.cls_token,
                                                      sep_token=tokenizer.sep_token,
                                                      cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                      # cls segment id
                                                      pad_on_left=bool(args.model_type in ['xlnet']),
                                                      # pad on the left for xlnet
                                                      pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_lengths = torch.tensor([f.length for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lengths, all_label_ids)
    return dataset


def load_and_cache_normal_example(args, word2index, type):
    task_class = processors[args.task_name]()
    # file: cached_train_bert-base-uncased_256_imdb
    file = 'normal_cached_{}_{}_{}'.format(
        type,
        args.max_seq_length,
        args.task_name
    )
    cached_features_file = os.path.join(args.cache_dir, file)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        pad_token = 0
        if type == 'train':
            examples = task_class.get_train_examples(args.data_dir)
        elif type == 'dev':
            examples = task_class.get_dev_examples(args.data_dir)
        else:
            examples = task_class.get_test_examples(args.data_dir)
        max_seq_length = args.max_seq_length//2

        labels = []
        x1_list = []
        x1_mask_list = []
        x1_lengths = []
        x2_list = []
        x2_mask_list = []
        x2_lengths = []
        for example in examples:
            text_a = example.text_a.lower()
            tokens_a = word_tokenize(text_a)
            if len(tokens_a) > max_seq_length:
                tokens_a = tokens_a[:max_seq_length]
            x1 = [word2index.get(token, word2index['<unk>']) for token in tokens_a]
            padding_length = max_seq_length - len(x1)
            x1_lengths.append(len(x1))
            x1_mask = [1] * len(x1)
            x1 = x1 + ([pad_token] * padding_length)
            x1_mask = x1_mask + ([0] * padding_length)

            text_b = example.text_b.lower()
            tokens_b = word_tokenize(text_b)
            if len(tokens_b) > max_seq_length:
                tokens_b = tokens_b[:max_seq_length]
            x2 = [word2index.get(token, word2index['<unk>']) for token in tokens_b]
            padding_length = max_seq_length - len(x2)
            x2_lengths.append(len(x2))
            x2_mask = [1] * len(x2)
            x2 = x2 + ([pad_token] * padding_length)
            x2_mask = x2_mask + ([0] * padding_length)

            labels.append(example.label)
            x1_list.append(x1)
            x1_mask_list.append(x1_mask)
            x2_list.append(x2)
            x2_mask_list.append(x2_mask)
        features = (x1_list, x1_mask_list, x2_list, x2_mask_list, labels, x1_lengths, x2_lengths)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    x1_ids = torch.tensor(features[0], dtype=torch.long)
    x1_mask = torch.tensor(features[1], dtype=torch.long)
    x2_ids = torch.tensor(features[2], dtype=torch.long)
    x2_mask = torch.tensor(features[3], dtype=torch.long)
    label_ids = torch.tensor(features[4], dtype=torch.long)
    x1_lengths = torch.tensor(features[5], dtype=torch.long)
    x2_lengths = torch.tensor(features[6], dtype=torch.long)

    dataset = TensorDataset(x1_ids, x1_mask, x2_ids, x2_mask, label_ids, x1_lengths, x2_lengths)
    return dataset


def _predict(model, model_type, batch):
    if model_type == 'bert':
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[4]}
        outputs = model(**inputs)
    elif model_type in ['bow', 'decom_att']:
        inputs = {'x1': batch[0], 'x1_mask': batch[1], 'x2': batch[2], 'x2_mask': batch[3],
                  'labels': batch[4]}
        outputs = model(**inputs)
    elif model_type == 'esim':
        inputs = {'premises': batch[0], 'premises_lengths': batch[5], 'hypotheses': batch[2], 'hypotheses_lengths': batch[6],
                  'labels': batch[4]}
        outputs = model(**inputs)
    else:
        outputs = None
    return outputs


def save(model_type, model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'checkpoint.pth')
    if model_type == 'bert':
        model_to_save = model.module if hasattr(model, 'module') else model
        config_file = os.path.join(output_dir, 'config.bin')
        torch.save(model.state_dict(), output_file)
        model_to_save.config.to_json_file(config_file)
    else:
        torch.save(model.state_dict(), output_file)


def load(args, checkpoint_dir):
    state_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            namekey = k[7:]  # remove `module.`
        else:
            namekey = k
        new_state_dict[namekey] = v

    if args.model_type == 'bert':
        config = BertConfig.from_json_file(os.path.join(checkpoint_dir, 'config.bin'))
        model = BertForSequenceClassification(config)
        model.load_state_dict(new_state_dict)
    elif args.model_type == 'bow':
        model = BOWModel(new_state_dict['embedding.weight'], n_vocab=args.vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size, num_classes=args.num_labels)
        model.load_state_dict(new_state_dict)
    elif args.model_type == 'decom_att':
        model = DecompAttentionModel(None, n_vocab=args.vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size, num_classes=args.num_labels)
        model.load_state_dict(new_state_dict)
    elif args.model_type == 'esim':
        model = ESIM(vocab_size=args.vocab_size, embedding_dim=args.embed_size, hidden_size=args.hidden_size, embeddings=None,
                 padding_idx=0,
                 dropout=0.1,
                 num_classes=args.num_labels,
                 device=args.device)
        model.load_state_dict(new_state_dict)
    else:
        raise ValueError('model type is not found!')

    return model.to(args.device)


def normal_train(args, model, train_dataset, eval_dataset, start_epoch=0):
    '''
    normal train
    :param start_epoch:
    :param eval_dataset:
    :param args:
    :param train_dataset:
    :param model:
    :param tokenizer:
    :return:
    '''
    batch_size = args.batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    # Prepare optimizer and schedule (linear warmup and decay)
    if args.model_type == 'bert':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs, eta_min=0)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  batch size per GPU = %d", args.batch_size)

    train_steps = 0
    train_loss = 0.0
    best_acc = 0.
    best_loss = 10000.

    for e in range(start_epoch, args.num_train_epochs):
        scheduler.step(e)
        for batch in tqdm.tqdm(train_dataloader):
            # print(batch[0].shape)
            # print(batch[1].shape)
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            outputs = _predict(model, args.model_type, batch)
            loss = outputs[0]
            train_steps += 1
            train_loss += loss.item()
            if train_steps % 1000 == 0:
                print('loss=', train_loss / train_steps)
            if train_steps % 10000 == 0 and train_loss / train_steps < best_loss:
                best_loss = train_loss / train_steps
                output_dir = os.path.join(args.output_dir,
                                          'normal_{}_{}_checkpoint-{}_'.format(args.task_name, args.model_type, e))
                save(args.model_type, model, output_dir)
                logger.info("Saving model checkpoint to {}".format(output_dir))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

        # save model.
        output_dir = os.path.join(args.output_dir,
                                  'normal_{}_{}_checkpoint-{}'.format(args.task_name, args.model_type, e))
        acc = evaluate(args, model, eval_dataset)
        logger.info("epoch={}, acc={:.4f}".format(e, acc))

        if args.model_type in ['bert', 'xlnet', 'xlm']:
            save(args.model_type, model, output_dir)
            logger.info("Saving model checkpoint to {}".format(output_dir))
        elif acc > best_acc:
            best_acc = acc
            save(args.model_type, model, output_dir)
            logger.info("Saving model checkpoint to {}".format(output_dir))

    return train_steps, train_loss / train_steps


def evaluate(args, model, eval_dataset):
    batch_size = args.batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    logger.info("***** Running normal evaluation *****")
    logger.info(" Num examples = %d", len(eval_dataset))
    logger.info(" Batch size = %d", batch_size)
    eval_loss = 0.
    eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm.tqdm(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            outputs = _predict(model, args.model_type, batch)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += np.mean(tmp_eval_loss.tolist())
        eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch[4].detach().cpu().numpy()
        else:
            label_ids = batch[4].detach().cpu().numpy()
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # (B, 2)
            out_label_ids = np.append(out_label_ids, label_ids, axis=0)  # (B,)
    preds = np.argmax(preds, axis=1)
    acc = utils.compute_metrics(preds, out_label_ids)
    logger.info("eval result acc={:.4f} loss={:.2f}".format(acc, eval_loss / eval_steps))

    return acc


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--data_dir', default=None, type=str, required=True, help="The input data dir.")
    parser.add_argument('--model_type', default=None, type=str, required=True,
                        help="Model type selected in [bert, xlnet, xlm, bow, decom_att]")
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str,
                        help="Shortcut name is selected in [bert-base-uncased, ]")
    parser.add_argument('--task_name', default='snli', type=str, help="The name of task is selected in [snli]")
    parser.add_argument('--output_dir', default='../out', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # other parameters
    parser.add_argument("--cache_dir", default='../cache', type=str, help="Store the cache files.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm. Avoiding over-fitting.")
    parser.add_argument("--num_train_epochs", default=60, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initializaiton.")
    parser.add_argument("--train", action='store_true', help="Whether to run training.")
    parser.add_argument("--eval", action='store_true', help="Whether to run eval on dev set.")
    parser.add_argument("--ckpt", default=-1, type=int, help="Which ckpt to load.")
    parser.add_argument("--from_scratch", action='store_true', help="Whether to train from scratch.")
    parser.add_argument("--train_type", default='normal', type=str, help="Train type is selected in [normal, rs].")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError("input data dir is not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("model type: %s, task name: %s, device: %s, ", args.model_type, args.task_name, device)

    # set seed
    set_seed(args)
    # Prepare task
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)

    task_class = processors[args.task_name]()
    label_list = task_class.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels
    # load vocab.
    if args.model_type != 'bert':
        if os.path.exists(args.cache_dir + '/{}_vocab_train.pkl'.format(args.task_name)):
            with open(args.cache_dir + '/{}_vocab_train.pkl'.format(args.task_name), 'rb') as f:
                vocab = pickle.load(f)
            index2word = vocab['index2word']
            word2index = vocab['word2index']
            word_mat = vocab['word_mat']
        else:
            glove_path = '../data/glove/glove.840B.300d.txt'
            index2word, word2index, word_mat = load_vocab(args, glove_path)
        args.word_mat = word_mat
        args.vocab_size = len(index2word)

    # load model.
    model = None
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
        args.vocab_size = tokenizer.vocab_size
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels,
                                            finetuning_task=args.task_name)
        model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_type == 'bow':
        args.embed_size = 300
        args.hidden_size = 100
        model = BOWModel(word_mat, n_vocab=args.vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size, num_classes=args.num_labels)
    elif args.model_type == 'decom_att':  # No using
        args.embed_size = 300
        args.hidden_size = 100
        model = DecompAttentionModel(word_mat, n_vocab=args.vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size, num_classes=args.num_labels)
    elif args.model_type == 'esim':
        args.embed_size = 300
        args.hidden_size = 100
        model = ESIM(vocab_size=args.vocab_size, embedding_dim=args.embed_size, hidden_size=args.hidden_size, embeddings=torch.tensor(word_mat).float(),
                 padding_idx=0,
                 dropout=0.1,
                 num_classes=args.num_labels,
                 device=args.device)
    else:
        raise ValueError('model type is not found!')
    model.to(device)
    logger.info("Training/evaluation parameters %s", args)

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Create cache directory if needed
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if args.train:
        train_dataset = load_and_cache_normal_example(args, word2index, 'train') if args.model_type not in ['bert'] else load_and_cache_bert_example(args, tokenizer, 'train')
        eval_dataset = load_and_cache_normal_example(args, word2index, 'eval') if args.model_type not in ['bert'] else load_and_cache_bert_example(args, tokenizer, 'eval')
    if args.eval:
        test_dataset = load_and_cache_normal_example(args, word2index, 'test') if args.model_type not in ['bert'] else load_and_cache_bert_example(args, tokenizer, 'test')

    # Training
    if args.train:
        if args.from_scratch:  # default False
            global_step, train_loss = normal_train(args, model, train_dataset, eval_dataset)
        else:
            if args.ckpt < 0:
                checkpoints = glob.glob(
                    args.output_dir + '/normal_{}_{}_checkpoint-*'.format(args.task_name, args.model_type))
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
                checkpoint = checkpoints[-1]
                ckpt = int(checkpoint.split('-')[-1])
            else:
                checkpoint = os.path.join(args.output_dir, 'normal_{}_{}_checkpoint-{}'.format(args.task_name, args.model_type, args.ckpt))
                ckpt = args.ckpt
            model = load(args, checkpoint)
            print("Load model from {}".format(checkpoint))
            global_step, train_loss = normal_train(args, model, train_dataset, eval_dataset, ckpt + 1)
        logger.info(" global_step = %s, average loss = %s", global_step, train_loss)

    # Evaluation
    if args.eval:
        if args.ckpt < 0:
            checkpoints = glob.glob(
                args.output_dir + '/{}_{}_{}_checkpoint-*'.format(args.train_type, args.task_name, args.model_type))
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            checkpoint = checkpoints[-1]
        else:
            checkpoint = os.path.join(args.output_dir, '{}_{}_{}_checkpoint-{}'.format(args.train_type, args.task_name, args.model_type, args.ckpt))
        model = load(args, checkpoint)
        print("Evaluation result, load model from {}".format(checkpoint))
        acc = evaluate(args, model, test_dataset)
        print("acc={:.4f}".format(acc))


if __name__ == '__main__':
    main()