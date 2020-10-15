# -*- coding: utf-8 -*-#
# Name:         evaluation
# Description:  
# Author:       Fali Wang
# Date:         2020/8/28 21:48

import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
import pickle
import json
import string

import utils
from utils import InputExample, WordSubstitute
from model import CNNModel, LSTMModel, CharCNN
from imdb_dataset import ImdbProcessor
from amazon_dataset import AmazonProcessor

logger = logging.getLogger(__name__)

processors = {
    "imdb": ImdbProcessor,
    "amazon": AmazonProcessor,
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def _create_skipped_testset(path_of_data, folder, skip):
    '''
    data from folder with skip, save to path_of_data
    :param path_of_data:
    :param folder:
    :param skip:
    :return:
    '''
    if not os.path.exists(path_of_data):
        os.makedirs(path_of_data)

    path_list = os.listdir(folder)
    path_list.sort()
    count = 0
    for filename in tqdm.tqdm(path_list):
        if count % skip == 0:
            x_raw = open(os.path.join(folder, filename)).read()
            with open(os.path.join(path_of_data, filename), 'w') as a:
                a.write(x_raw)
        count += 1


def _randomized_create_examples_from_folder(folder, label):
    examples = []
    i = 0
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r') as f:
            tem_text = f.readlines()
            if tem_text:
                text_a = tem_text[0]
                guid = "%s-%d" % ('test', i)
                i += 1
                text_b = None
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def _create_folder_randomized_test_set(output_data_dir, folder, num_random_sample, random_smooth, label):
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    path_list = os.listdir(folder)
    print('tqdm _create_folder_randomized_test_set...')
    for filename in tqdm.tqdm(path_list):
        data_name = os.path.splitext(filename)[0]  # name, example 1.txt -> 1
        data = open(os.path.join(folder, filename)).read()
        if data:
            path_of_data = os.path.join(output_data_dir, data_name)
            if not os.path.exists(path_of_data):
                os.makedirs(path_of_data)
            # 以每一个原始样本的文件名为目录名创建文件夹，随机采样num_random_sample个样本
            for _ in range(num_random_sample):
                data_perturb = str(random_smooth.get_perturbed_batch(np.array([[data]]))[0][0])
                with open(os.path.join(path_of_data, data_name + '_' + str(_) + '.txt'), 'w') as a:
                    a.write(data_perturb)

            examples = _randomized_create_examples_from_folder(path_of_data, label)
            torch.save(examples, path_of_data + '/example')


def _randomize_imdb_testset(data_dir, skip, similarity_threshold, perturbation_constraint, num_random_sample, random_smooth):

    pos_folder = os.path.join(data_dir, 'test', 'pos')
    neg_folder = os.path.join(data_dir, 'test', 'neg')

    # creating the skipped test set
    pos_skip_folder = os.path.join(data_dir, 'test', 'pos' + str(skip))
    _create_skipped_testset(pos_skip_folder, pos_folder, skip)
    neg_skip_folder = os.path.join(data_dir, 'test', "neg" + str(skip))
    _create_skipped_testset(neg_skip_folder, neg_folder, skip)

    # creating folder for randomized testset
    print('Generating randomized data for pos label.')
    out_data_dir = os.path.join(data_dir,
                                'random_test_' + str(similarity_threshold) + '_' + str(perturbation_constraint),
                                "pos" + str(skip))
    _create_folder_randomized_test_set(out_data_dir, pos_skip_folder, num_random_sample, random_smooth, 1)
    print('Generating randomized data for neg label.')
    out_data_dir = os.path.join(data_dir,
                                'random_test_' + str(similarity_threshold) + '_' + str(perturbation_constraint),
                                "neg" + str(skip))
    _create_folder_randomized_test_set(out_data_dir, neg_skip_folder, num_random_sample, random_smooth, 0)


def _randomize_amazon_testset(data_dir, skip, similarity_threshold, perturbation_constraint, num_random_sample, random_smooth):
    processor = AmazonProcessor()
    examples = processor.get_dev_examples(data_dir)

    random_skip_folder = os.path.join(data_dir, 'random_test_' + str(similarity_threshold) + '_' + str(perturbation_constraint))
    skip_folder = os.path.join(data_dir, 'test'+str(skip))
    if not os.path.exists(random_skip_folder):
        os.makedirs(random_skip_folder)
        os.makedirs(skip_folder)
    print('Generating randomized data.')
    for i, example in enumerate(examples):
        if i % skip == 0:
            data = example.text_a
            label = example.label
            # i is the folder name
            path_of_data = os.path.join(random_skip_folder, str(i))
            if not os.path.exists(path_of_data):
                os.makedirs(path_of_data)
            origin_file = os.path.join(skip_folder, str(i) + '.txt')
            with open(origin_file, 'w') as a:
                a.write(data)
            for _ in range(num_random_sample):
                data_perturb = str(random_smooth.get_perturbed_batch(np.array([[data]]))[0][0])
                with open(os.path.join(path_of_data, str(i) + '_' + str(_) + '.txt'), 'w') as a:
                    # label,perturb_text
                    a.write(data_perturb)  # str(label) + ',' +
            examples = _randomized_create_examples_from_folder(path_of_data, label)
            torch.save(examples, path_of_data + '/example')


def randomize_testset(args, random_smooth, similarity_threshold, perturbation_constraint):
    data_dir = args.data_dir
    skip = args.skip
    num_random_sample = args.num_random_sample
    if os.path.exists(
            os.path.join(data_dir, 'random_test_' + str(similarity_threshold) + '_' + str(perturbation_constraint))):
        return
    if args.task_name == 'imdb':
        _randomize_imdb_testset(data_dir, skip, similarity_threshold, perturbation_constraint, num_random_sample, random_smooth)
    elif args.task_name == 'amazon':
        _randomize_amazon_testset(data_dir, skip, similarity_threshold, perturbation_constraint, num_random_sample, random_smooth)


def calculate_tv_perturb(args, perturb):
    '''
    calcalate total variation(tv), namely q_x.
    :param args:
    :param perturb:
    :return:
    '''
    similarity_threshold = args.similarity_threshold
    cache_dir = args.cache_dir
    task_name = args.task_name

    counterfitted_tv_file = os.path.join(cache_dir,
                                         task_name + '_counterfitted_tv_pca' + str(similarity_threshold) + '_' + str(
                                             args.perturbation_constraint) + '.pkl')
    if os.path.exists(counterfitted_tv_file):
        return

    # reading vocabulary
    with open(os.path.join(cache_dir, task_name + '_vocab_pca.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    # reading neighbor set
    with open(os.path.join(cache_dir, task_name + '_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'),
              'rb') as f:
        neighbor = pickle.load(f)
    neighbor = neighbor['neighbor']

    total_intersect = 0
    total_freq = 0
    counterfitted_tv = {}  # word:q_x
    print('tqdm calculate_tv_perturb...')
    for key, _ in tqdm.tqdm(neighbor.items()):
        if not key in perturb:
            counterfitted_tv[key] = 1
            total_intersect += vocab[key]['freq'] * 1
            total_freq += vocab[key]['freq']
        elif perturb[key]['isdivide'] == 0:
            counterfitted_tv[key] = 1
            total_intersect += vocab[key]['freq'] * 1
            total_freq += vocab[key]['freq']
        else:
            key_neighbor = neighbor[key]
            cur_min = 1.
            num_perb = len(perturb[key]['set'])
            for neig in key_neighbor:
                num_inter_perb = len(list(set(perturb[neig]['set']).intersection(set(perturb[key]['set']))))
                tem_min = num_inter_perb / num_perb
                if tem_min < cur_min:
                    cur_min = tem_min
            counterfitted_tv[key] = cur_min
            total_intersect += vocab[key]['freq'] * cur_min
            total_freq += vocab[key]['freq']

    with open(counterfitted_tv_file, 'wb') as f:
        pickle.dump(counterfitted_tv, f)
    print('calculate total variation finishes.')


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
    elif args.model_type == 'cnn':
        model = CNNModel(n_vocab=args.vocab_size, embed_size=args.embed_size, num_classes=args.num_labels,
                         num_filters=args.num_filters, filter_sizes=args.filter_sizes, device=args.device)
        model.load_state_dict(new_state_dict)
    elif args.model_type == 'lstm':
        model = LSTMModel(n_vocab=args.vocab_size, embed_size=args.embed_size, num_classes=args.num_labels,
                          hidden_size=args.hidden_size, device=args.device)
        model.load_state_dict(new_state_dict)
    elif args.model_type == 'char-cnn':
        model = CharCNN(num_features=args.num_features, num_classes=args.num_labels)
        model.load_state_dict(new_state_dict)
    else:
        raise ValueError('model type is not found!')

    return model.to(args.device)


def _get_tv(text, counterfitted_tv, key_set, exclude):
    # exclude = set(string.punctuation)
    # key_set = set(counterfitted_tv.keys())
    tokens = text.split(' ')
    tv_list = np.zeros(len(tokens))
    for i, token in enumerate(tokens):
        if token[-1] in exclude:
            token = token[:-1]

        if token in key_set:
            tv_list[i] = counterfitted_tv[token]
        else:
            tv_list[i] = 1.
    return np.sort(tv_list)


def _load_and_cache_examples(args, folder, task, tokenizer):
    # folder is each text's folder
    task_class = processors[task]()
    if args.model_type != 'char-cnn':
        cached_features_file = os.path.join(folder, 'cached_features_{}_{}_{}_{}_{}'.format(
            'test',
            args.model_name_or_path,
            args.max_seq_length,
            args.task_name,
            args.similarity_threshold
        ))
        if os.path.exists(cached_features_file):
            features = torch.load(cached_features_file)
        else:
            label_list = task_class.get_labels()
            examples = torch.load(folder + '/example')
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
            torch.save(features, cached_features_file)
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_lengths = torch.tensor([f.length for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_lengths)
        return dataset
    else:
        file = 'cached_char-cnn_{}_{}_{}_{}_dataset'.format(
            'test',
            args.model_type,
            args.l0,
            args.task_name
        )
        cached_dataset_file = os.path.join(folder, file)
        if os.path.exists(cached_dataset_file):
            dataset = torch.load(cached_dataset_file)
        else:
            # logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = task_class.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}
            examples = torch.load(folder + '/example')
            X = []
            labels = []
            for example in tqdm.tqdm(examples):
                text_a = example.text_a.lower()
                x = torch.zeros(args.num_features, args.l0)
                y = label_map[example.label]
                for i, char in enumerate(text_a[::-1]):
                    if i == args.l0:
                        break
                    if args.alphabets.find(char) != -1:
                        x[args.alphabets.find(char)][i] = 1.
                X.append(x.numpy())  # do not convert to list, will lead to ValueError: only one element tensors can be
                # converted to Python scalars
                labels.append(y)
                # if ind == 15:
                #     break
            x = torch.tensor(X, dtype=torch.float)  # 7G RAM
            y = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(x, y)
            # logger.info("Saving features into cached file %s", cached_dataset_file)
            # torch.save(dataset, cached_dataset_file)
        return dataset


def _predict(model, model_type, batch):
    if model_type == 'bert':
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
        outputs = model(**inputs)
    elif model_type in ['cnn', 'lstm']:
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3],
                  'lengths': batch[4]}
        outputs = model(**inputs)
    else:
        inputs = {'x': batch[0], 'labels': batch[1]}
        outputs = model(**inputs)
    return outputs


def _certify_randomized_smooth(args, random_data_dir, tokenizer, model, counterfitted_tv, key_set, label_type):
    '''
    :param random_data_dir: [random_data_dir_pos or random_data_dir_neg]
    :return:
    '''
    similarity_threshold = args.similarity_threshold
    task_name = args.task_name

    text_count = 0
    results = {}
    total_cert_count = 0
    total_count = 0

    if os.path.exists(random_data_dir):
        files = os.listdir(random_data_dir)  # dirs
        print('tqdm evaluate each text ...', label_type)
        for file in tqdm.tqdm(files):  # file is a dir
            # read original text
            original_text_file = os.path.join(args.data_dir, 'test', label_type, file + '.txt')
            original_text = open(original_text_file).read()
            eval_dataset = _load_and_cache_examples(args, os.path.join(random_data_dir, file), task_name, tokenizer)
            batch_size = args.batch_size
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

            eval_loss = 0.
            eval_steps = 0
            preds = None
            preds_softmax = None
            out_label_ids = None
            for batch in tqdm.tqdm(eval_dataloader):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)
                with torch.no_grad():
                    outputs = _predict(model, args.model_type, batch)
                    tmp_eval_loss, logits = outputs[:2]
                    eval_loss += tmp_eval_loss.item()
                eval_steps += 1
                label_ids =  batch[3].detach().cpu().numpy() if args.model_type != 'char-cnn' else batch[1].detach().cpu().numpy()
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    preds_softmax = logits.softmax(dim=-1)
                    out_label_ids = label_ids
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # (B, 2)
                    preds_softmax = torch.cat((preds_softmax, logits.softmax(dim=-1)), dim=0)  # (B, 2)
                    out_label_ids = np.append(out_label_ids, label_ids)  # (B,)
            preds = np.argmax(preds, axis=1)
            preds_mean = preds_softmax.mean(dim=0)  # (2,)
            if args.train_type == 'normal':
                acc = utils.compute_metrics(preds, out_label_ids)
            else:  # rs
                label = out_label_ids[0].tolist()
                acc = preds_mean[label].item()  # scalar, item() and tolist() equally
            # logger.info("eval result acc={:.4f} loss={:.2f}".format(acc, eval_loss / eval_steps))
            results[file] = {'acc': acc, 'text': original_text, 'similarity_threshold': similarity_threshold,
                             'label': label_type}
            # q_x list, asc
            tem_tv = _get_tv(original_text, counterfitted_tv, key_set, set(string.punctuation))
            # calc certified bound ***
            # ???
            if acc - 1. + np.prod(tem_tv[:20]) >= 0.5 + args.mc_error:  # reduction
                total_cert_count += 1
            total_count += 1
            if text_count % 10 == 0:
                print(' certified acc:', total_cert_count / total_count)
            text_count += 1

    return total_count, total_cert_count, results


def _randomized_evaluate_imdb(args, tokenizer, model, counterfitted_tv, key_set, cache_dir):
    random_data_dir_pos = os.path.join(args.data_dir, 'random_test_' + str(args.similarity_threshold) + '_' + str(
        args.perturbation_constraint), 'pos' + str(args.skip))
    random_data_dir_neg = os.path.join(args.data_dir, 'random_test_' + str(args.similarity_threshold) + '_' + str(
        args.perturbation_constraint), 'neg' + str(args.skip))
    # pos certify
    total_count1, total_cert_count1, results1 = _certify_randomized_smooth(args, random_data_dir_pos, tokenizer, model,
                                                                           counterfitted_tv, key_set, 'pos')
    print('pos certify acc is {:.4f}'.format(total_cert_count1 / total_count1))
    # neg certify
    total_count2, total_cert_count2, results2 = _certify_randomized_smooth(args, random_data_dir_neg, tokenizer, model,
                                                                           counterfitted_tv, key_set, 'neg')
    print('neg certify acc is {:.4f}'.format(total_cert_count2 / total_count2))

    results = dict(results1, **results2)  # merge
    results_save_name = os.path.join(cache_dir, 'cached_results_{}_{}_{}_{}_{}_{}'.format(
        'test',
        args.model_type,  # list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name),
        str(args.similarity_threshold),
        str(args.perturbation_constraint)))
    with open(results_save_name, 'wb') as f:
        pickle.dump(results, f)
    print('certified acc: ', (total_cert_count1 + total_cert_count2) / (total_count1 + total_count2))


def _randomized_evaluate_amazon(args, tokenizer, model, counterfitted_tv, key_set):
    random_data_dir = os.path.join(args.data_dir, 'random_test_' + str(args.similarity_threshold) + '_' + str(args.perturbation_constraint))

    text_count = 0
    results = {}
    total_cert_count = 0
    total_count = 0

    for folder in tqdm.tqdm(os.listdir(random_data_dir)):
        original_file_path = os.path.join(args.data_dir, 'test'+str(args.skip), folder + '.txt')
        original_text = open(original_file_path, 'r').read()
        path_of_data = os.path.join(random_data_dir, folder)
        eval_dataset = _load_and_cache_examples(args, path_of_data, args.task_name, tokenizer)
        batch_size = args.batch_size
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

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
                eval_loss += tmp_eval_loss.item()
            eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # (B, 2)
                out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)  # (B,)
        preds = np.argmax(preds, axis=1)
        acc = utils.compute_metrics(preds, out_label_ids)
        # logger.info("eval result acc={:.4f} loss={:.2f}".format(acc, eval_loss / eval_steps))

        # q_x list, asc
        tem_tv = _get_tv(original_text, counterfitted_tv, key_set, set(string.punctuation))
        # calc certified bound ***
        # ???
        if acc - 1. + np.prod(tem_tv[:20]) >= 0.5 + args.mc_error:  # reduction
            total_cert_count += 1
        total_count += 1
        if text_count % 10 == 0:
            print(' certified acc:', total_cert_count / total_count)
        text_count += 1

    print('certified acc: ', total_cert_count / total_count)
    return total_count, total_cert_count, results


def randomized_evaluate(args, model, tokenizer):
    # read tv perturb
    similarity_threshold = args.similarity_threshold
    cache_dir = args.cache_dir
    task_name = args.task_name

    counterfitted_tv_file = os.path.join(cache_dir,
                                         task_name + '_counterfitted_tv_pca' + str(similarity_threshold) + '_' + str(
                                             args.perturbation_constraint) + '.pkl')
    with open(counterfitted_tv_file, 'rb') as f:
        counterfitted_tv = pickle.load(f)  # dict
        key_set = set(counterfitted_tv.keys())

    if task_name == 'imdb':
        _randomized_evaluate_imdb(args, tokenizer, model, counterfitted_tv, key_set, cache_dir)
    elif task_name == 'amazon':
        _randomized_evaluate_amazon(args, tokenizer, model, counterfitted_tv, key_set)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of task is selected in [imdb, amazon]")
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir.")
    parser.add_argument("--cache_dir", default='../cache', type=str, help="The cache data dir.")
    parser.add_argument('--model_type', default=None, type=str, required=True,
                        help="Model type selected in [bert, xlnet, xlm, cnn, lstm]")
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str,
                        help="Shortcut name is selected in [bert-base-uncased, ]")
    parser.add_argument('--output_dir', default='../out', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--skip", default=20, type=int, help="Evaluate one testing point every skip testing point.")
    parser.add_argument("--num_random_sample", default=5000, type=int,
                        help="The number of random samples of each texts.")
    parser.add_argument("--similarity_threshold", default=0.8, type=float, help="The similarity constraint to be "
                                                                                "considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int, help="The maximum size of perturbation "
                                                                                 "set of each word.")
    parser.add_argument("--mc_error", default=0.01, type=float,
                        help="Monte Carlo Error based on concentration inequality.")
    parser.add_argument("--train_type", default='normal', type=str, help="Train type is selected in [normal, rs].")
    # other parameters
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--ckpt", default=-1, type=int, help="Which ckpt to load.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initializaiton.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("model type: %s, task name: %s, device: %s, train_type: %s", args.model_type, args.task_name, device, args.train_type)

    set_seed(args)
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)

    task_class = processors[args.task_name]()
    label_list = task_class.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    args.vocab_size = tokenizer.vocab_size
    if args.model_type == 'bert':
        pass
        # config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels,
        #                                     finetuning_task=args.task_name)
        # model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_type == 'cnn':
        args.embed_size = 300
        args.num_filters = 100
        args.filter_sizes = (3,)
        # model = CNNModel(n_vocab=args.vocab_size, embed_size=args.embed_size, num_classes=num_labels,
        #                  num_filters=args.num_filters, filter_sizes=args.filter_sizes, device=args.device)
    elif args.model_type == 'lstm':
        args.embed_size = 300
        args.hidden_size = 100
        # model = LSTMModel(n_vocab=args.vocab_size, embed_size=args.embed_size, num_classes=num_labels,
        #                   hidden_size=args.hidden_size, device=args.device)
    elif args.model_type == 'char-cnn':
        args.alphabets = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"\\/|_@#$%^&*~`+-=<>()[]{}\n'
        args.num_features = len(args.alphabets)
        args.l0 = 1014
    else:
        raise ValueError('model type is not found!')
    # model.to(device)

    similarity_threshold = args.similarity_threshold
    perturbation_constraint = args.perturbation_constraint

    perturbation_file = args.cache_dir + '/' + args.task_name + '_perturbation_constraint_pca' + str(
        similarity_threshold) + "_" + str(perturbation_constraint) + '.pkl'
    with open(perturbation_file, 'rb') as f:
        perturb = pickle.load(f)
    # shorten the perturbation set tot desired length constraint
    # for k, _ in perturb.items():
    #     if len(perturb[k]['set']) > perturbation_constraint:
    #         perturb[k]['set'] = perturb[k]['set'][:perturbation_constraint]
    #         perturb[k]['isdivide'] = 1

    # random smooth
    random_smooth = WordSubstitute(perturb)
    # generate randomized data
    randomize_testset(args, random_smooth, similarity_threshold, perturbation_constraint)
    # calculate total variation
    calculate_tv_perturb(args, perturb)
    # Evaluation
    if args.ckpt < 0:
        checkpoints = glob.glob(
            args.output_dir + '/{}_{}_{}_checkpoint-*'.format(args.train_type, args.task_name, args.model_type))
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        checkpoint = checkpoints[-1]
    else:
        checkpoint = os.path.join(args.output_dir,
                                  '{}_{}_{}_checkpoint-{}'.format(args.train_type, args.task_name, args.model_type, args.ckpt))
    print("Evaluation result, load model from {}".format(checkpoint))
    model = load(args, checkpoint)
    randomized_evaluate(args, model, tokenizer)


if __name__ == '__main__':
    main()