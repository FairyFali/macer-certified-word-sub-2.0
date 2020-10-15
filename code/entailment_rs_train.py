# -*- coding: utf-8 -*-#
# Name:         entailment_rs_train
# Description:  
# Author:       Fali Wang
# Date:         2020/10/11 20:15
import argparse
import glob
import logging
import os
import random
import numpy as np
import tqdm
import string
import pickle
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from nltk import word_tokenize

import utils
from utils import InputExample, InputFeatures, WordSubstitute
from entailment_train_evaluate import _predict, save, load, evaluate, processors, load_and_cache_normal_example, load_and_cache_bert_example, load_vocab
from entailment_rs_evaluation import _get_tv
from model import *

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def random_smooth_train(args, model, task_class, data_dir, num_random_sample, random_smooth, model_type, tokenizer, word2index, gamma, counterfitted_tv, key_set, lbd, st=0):
    examples = task_class.get_train_examples(data_dir)
    # Prepare optimizer and schedule (linear warmup and decay)
    if args.model_type == 'bert':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs, eta_min=0)

    if args.model_type != 'bert':
        eval_dataset = load_and_cache_normal_example(args, word2index, 'eval')
    else:
        eval_dataset = load_and_cache_bert_example(args, tokenizer, 'eval')

    np.random.shuffle(examples)
    label_list = task_class.get_labels()
    iter_loss = 0.
    iter_loss_classifier = 0.
    iter_loss_robust = 0.
    iter_steps = 0
    best_acc = 0.
    for e in range(st, args.num_train_epochs):
        scheduler.step(e)
        for example in tqdm.tqdm(examples):
            prem = example.text_a.lower()
            hypo = example.text_b.lower()
            label = example.label
            random_examples = []
            for _ in range(num_random_sample):
                hypo_perturb = str(random_smooth.get_perturbed_batch(np.array([[hypo]]))[0][0])
                random_examples.append(
                    InputExample(None, prem, hypo_perturb, label)
                )
            label = torch.tensor([label], device=args.device)
            if model_type == 'bert':
                features = utils.convert_examples_to_features(random_examples, label_list, args.max_seq_length, tokenizer,
                                                              cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                              # xlnet has a cls token at the end
                                                              cls_token=tokenizer.cls_token,
                                                              sep_token=tokenizer.sep_token,
                                                              cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                              # cls segment id
                                                              pad_on_left=bool(args.model_type in ['xlnet']),
                                                              # pad on the left for xlnet
                                                              pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
                all_lengths = torch.tensor([f.length for f in features], dtype=torch.long)
                dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lengths, all_label_ids)
            else:
                max_seq_length = args.max_seq_length // 2
                pad_token = 0
                labels = []
                x1_list = []
                x1_mask_list = []
                x1_lengths = []
                x2_list = []
                x2_mask_list = []
                x2_lengths = []
                for random_example in random_examples:
                    text_a = random_example.text_a.lower()
                    tokens_a = word_tokenize(text_a)
                    if len(tokens_a) > max_seq_length:
                        tokens_a = tokens_a[:max_seq_length]
                    x1 = [word2index.get(token, word2index['<unk>']) for token in tokens_a]
                    padding_length = max_seq_length - len(x1)
                    x1_lengths.append(len(x1))
                    x1_mask = [1] * len(x1)
                    x1 = x1 + ([pad_token] * padding_length)
                    x1_mask = x1_mask + ([0] * padding_length)

                    text_b = random_example.text_b.lower()
                    tokens_b = word_tokenize(text_b)
                    if len(tokens_b) > max_seq_length:
                        tokens_b = tokens_b[:max_seq_length]
                    x2 = [word2index.get(token, word2index['<unk>']) for token in tokens_b]
                    padding_length = max_seq_length - len(x2)
                    x2_lengths.append(len(x2))
                    x2_mask = [1] * len(x2)
                    x2 = x2 + ([pad_token] * padding_length)
                    x2_mask = x2_mask + ([0] * padding_length)

                    labels.append(random_example.label)
                    x1_list.append(x1)
                    x1_mask_list.append(x1_mask)
                    x2_list.append(x2)
                    x2_mask_list.append(x2_mask)
                features = (x1_list, x1_mask_list, x2_list, x2_mask_list, labels, x1_lengths, x2_lengths)
                x1_ids = torch.tensor(features[0], dtype=torch.long)
                x1_mask = torch.tensor(features[1], dtype=torch.long)
                x2_ids = torch.tensor(features[2], dtype=torch.long)
                x2_mask = torch.tensor(features[3], dtype=torch.long)
                label_ids = torch.tensor(features[4], dtype=torch.long)
                x1_lengths = torch.tensor(features[5], dtype=torch.long)
                x2_lengths = torch.tensor(features[6], dtype=torch.long)

                dataset = TensorDataset(x1_ids, x1_mask, x2_ids, x2_mask, label_ids, x1_lengths, x2_lengths)

            batch_size = args.batch_size
            train_sampler = RandomSampler(dataset)
            train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

            example_outputs = None
            i = 0
            for batch in train_dataloader:
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                outputs = _predict(model, args.model_type, batch)
                out = outputs[1]  # (B, classes_num)
                out = out.softmax(dim=-1)
                if example_outputs is None:
                    example_outputs = out
                else:
                    example_outputs = torch.cat((example_outputs, out), dim=0)  # (pop_size, classes_num)
                i += 1
            out_mean = example_outputs.mean(dim=0, keepdim=True)  # (1, classes_num)
            out_log_softmax = torch.log(out_mean + 1e-10)
            loss_classifier = F.nll_loss(out_log_softmax, label)

            p = out_mean[:, label[0]][0]
            p2 = out_mean.topk(2)[0][0, 0]
            if out_mean.topk(2)[1][0, 0].item() == label.item():
                p2 = out_mean.topk(2)[0][0, 1]
            tem_tv = _get_tv(hypo, counterfitted_tv, key_set, set(string.punctuation))
            delta_x = gamma - (p - p2 - 2. + 2*np.prod(tem_tv[:20]) - args.mc_error)  # binary classification
            loss_robust = delta_x

            pred = out_mean.argmax(dim=-1).item()
            if pred == label.item() and delta_x > 0:  # and iter_steps > 10000 # and delta_x > 0:
                loss = loss_classifier + lbd*loss_robust
            else:
                loss = loss_classifier
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

            iter_loss += loss.item()
            iter_loss_classifier += loss_classifier.item()
            iter_loss_robust += loss_robust.item()
            iter_steps += 1
            if iter_steps % 1000 == 0:
                print('====loss:', iter_loss/iter_steps, 'loss_classifier:', iter_loss_classifier/iter_steps, 'loss_robust:', iter_loss_robust/iter_steps)

            if iter_steps % 10000 == 0:
                acc = evaluate(args, model, eval_dataset)
                if acc > best_acc:
                    best_acc = acc
                    output_dir = os.path.join(args.output_dir,
                                              'rs_{}_{}_checkpoint-{}_'.format(args.task_name, args.model_type, e))
                    save(args.model_type, model, output_dir)
                    logger.info("Saving model checkpoint to {}".format(output_dir))

        # save model.
        output_dir = os.path.join(args.output_dir,
                                  'rs_{}_{}_checkpoint-{}'.format(args.task_name, args.model_type, e))
        save(args.model_type, model, output_dir)
        logger.info("Saving model checkpoint to {}".format(output_dir))

        evaluate(args, model, eval_dataset)
        print('Normal evaluation, not random smooth evaluation.')

    return iter_steps, iter_loss/iter_steps


def model_to_dataparallel(model, device, local_rank, n_gpu):
    model.to(device)
    if local_rank != -1:
        pass
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--data_dir', default=None, type=str, required=True, help="The input data dir.")
    parser.add_argument('--model_type', default=None, type=str, required=True,
                        help="Model type selected in [bert, xlnet, xlm, cnn, lstm]")
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str,
                        help="Shortcut name is selected in [bert-base-uncased, ]")
    parser.add_argument('--task_name', default=None, type=str, required=True,
                        help="The name of task is selected in [imdb, amazon]")
    parser.add_argument('--output_dir', default='../out', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--pop_size', default=32, type=int, help='Random smooth train require sample pop_size examples.')
    parser.add_argument('--lbd', default=1.0, type=float, help='Factor between classifier loss and robust loss.')
    parser.add_argument('--gamma', default=0.0, type=float, help='Parameter of hinge loss.')
    parser.add_argument("--similarity_threshold", default=0.8, type=float, help="The similarity constraint to be "
                                                                                "considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int, help="The maximum size of perturbation "
                                                                                 "set of each word.")
    parser.add_argument("--mc_error", default=0.01, type=float,
                        help="Monte Carlo Error based on concentration inequality.")
    # other parameters
    parser.add_argument("--cache_dir", default='../cache', type=str, help="Store the cache files.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError("input data dir is not exist.")

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("random smooth train, model type: %s, task name: %s, device: %s", args.model_type, args.task_name, device)

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
    word2index = None
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
    tokenizer = None
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
        args.vocab_size = tokenizer.vocab_size
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels,
                                            finetuning_task=args.task_name)
        model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_type == 'bow':
        args.embed_size = 300
        args.hidden_size = 100
        model = BOWModel(word_mat, n_vocab=args.vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size,
                         num_classes=args.num_labels)
    elif args.model_type == 'decom_att':  # No using
        args.embed_size = 300
        args.hidden_size = 100
        model = DecompAttentionModel(word_mat, n_vocab=args.vocab_size, embed_size=args.embed_size,
                                     hidden_size=args.hidden_size, num_classes=args.num_labels)
    elif args.model_type == 'esim':
        args.embed_size = 300
        args.hidden_size = 100
        model = ESIM(vocab_size=args.vocab_size, embedding_dim=args.embed_size, hidden_size=args.hidden_size,
                     embeddings=torch.tensor(word_mat).float(),
                     padding_idx=0,
                     dropout=0.1,
                     num_classes=args.num_labels,
                     device=args.device)
    else:
        raise ValueError('model type is not found!')
    model = model_to_dataparallel(model, device, args.local_rank, args.n_gpu)

    logger.info("Training/evaluation parameters %s", args)

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Create cache directory if needed
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # read tv perturb
    similarity_threshold = args.similarity_threshold
    cache_dir = args.cache_dir
    task_name = args.task_name

    counterfitted_tv_file = os.path.join(cache_dir,
                                         task_name + '_counterfitted_tv_pca' + str(
                                             similarity_threshold) + '_' + str(
                                             args.perturbation_constraint) + '.pkl')
    with open(counterfitted_tv_file, 'rb') as f:
        counterfitted_tv = pickle.load(f)  # dict
        key_set = set(counterfitted_tv.keys())

    # random smooth
    perturbation_file = args.cache_dir + '/' + args.task_name + '_perturbation_constraint_pca' + str(
        similarity_threshold) + "_" + str(args.perturbation_constraint) + '.pkl'
    with open(perturbation_file, 'rb') as f:
        perturb = pickle.load(f)
    random_smooth = WordSubstitute(perturb)

    if args.from_scratch:  # default False
        global_step, train_loss = random_smooth_train(args, model, task_class, args.data_dir, args.pop_size, random_smooth, args.model_type, tokenizer, word2index, args.gamma, counterfitted_tv, key_set, args.lbd)
    else:
        if args.ckpt < 0:
            checkpoints = glob.glob(
                args.output_dir + '/rs_{}_{}_checkpoint-*'.format(args.task_name, args.model_type))
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            checkpoint = checkpoints[-1]
            ckpt = int(checkpoint.split('-')[-1])
        else:
            checkpoint = os.path.join(args.output_dir,
                                      'rs_{}_{}_checkpoint-{}'.format(args.task_name, args.model_type, args.ckpt))
            ckpt = args.ckpt
        model = load(args, checkpoint)
        model = model_to_dataparallel(model, device, args.local_rank, args.n_gpu)
        print("Load model from {}".format(checkpoint))
        global_step, train_loss = random_smooth_train(args, model, task_class, args.data_dir, args.pop_size, random_smooth, args.model_type, tokenizer, word2index, args.gamma, counterfitted_tv, key_set, args.lbd, ckpt+1)
    logger.info(" global_step = %s, average loss = %s", global_step, train_loss)


if __name__ == '__main__':
    main()
