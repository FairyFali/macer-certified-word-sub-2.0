# -*- coding: utf-8 -*-#
# Name:         entailment_rs_evaluation
# Description:  
# Author:       Fali Wang
# Date:         2020/10/10 20:27

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
from nltk import word_tokenize

import utils
from utils import InputExample, WordSubstitute
from model import BOWModel, ESIM, DecompAttentionModel
from entailment import SNLIProcessor, EntailmentLabels

logger = logging.getLogger(__name__)

processors = {
    "snli": SNLIProcessor,
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def _create_skipped_testset(path_of_data, test_data_path, skip):
    '''
    data from folder with skip, save to path_of_data
    :param path_of_data:
    :param folder: original
    :param skip:
    :return:
    '''
    if not os.path.exists(path_of_data):
        os.makedirs(path_of_data)

    with open(test_data_path, 'r') as fn:
        for count, line in enumerate(fn):
            if count % skip == 0:
                data = json.loads(line)
                prem, hypo, label = data['sentence1'], data['sentence2'], data['gold_label']
                try:
                    label = EntailmentLabels[label].value
                except:
                    continue
                with open(os.path.join(path_of_data, str(count) + '.txt'), 'w') as a:
                    a.write(prem + '\t' + hypo + '\t' + str(label))


def _randomized_create_examples_from_folder(folder):
    examples = []
    i = 0
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), 'r') as f:
            tem_text = f.readlines()
            if tem_text:
                data = tem_text[0]
                prem, hypo, label = data.split('\t')
                guid = "%s-%d" % ('test', i)
                i += 1
                examples.append(
                    InputExample(guid=guid, text_a=prem, text_b=hypo, label=int(label)))
    return examples


def _create_folder_randomized_test_set(output_data_dir, folder, num_random_sample, random_smooth):
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    path_list = os.listdir(folder)  # folder is the path_of_data
    print('tqdm _create_folder_randomized_test_set...')
    for filename in tqdm.tqdm(path_list):
        data_name = os.path.splitext(filename)[0]  # name, example 1.txt -> 1
        data = open(os.path.join(folder, filename)).read()
        if data:
            prem, hypo, label = data.split('\t')
            path_of_data = os.path.join(output_data_dir, data_name)
            if not os.path.exists(path_of_data):
                os.makedirs(path_of_data)
            # 以每一个原始样本的文件名为目录名创建文件夹，随机采样num_random_sample个样本
            for _ in range(num_random_sample):
                hypo_perturb = str(random_smooth.get_perturbed_batch(np.array([[hypo]]))[0][0])
                with open(os.path.join(path_of_data, data_name + '_' + str(_) + '.txt'), 'w') as a:
                    a.write(prem + '\t' + hypo_perturb + '\t' + str(label))

            examples = _randomized_create_examples_from_folder(path_of_data)
            torch.save(examples, path_of_data + '/example')


def _randomize_snli_testset(data_dir, skip, similarity_threshold, perturbation_constraint, num_random_sample, random_smooth):

    test_file = os.path.join(data_dir, 'snli_1.0_test.jsonl')

    # creating the skipped test set
    skip_folder = os.path.join(data_dir, 'test')
    _create_skipped_testset(skip_folder, test_file, skip)

    # creating folder for randomized testset
    print('Generating randomized data.')
    out_data_dir = os.path.join(data_dir,
                                'random_test_' + str(similarity_threshold) + '_' + str(perturbation_constraint))
    _create_folder_randomized_test_set(out_data_dir, skip_folder, num_random_sample, random_smooth)


def randomize_testset(args, random_smooth, similarity_threshold, perturbation_constraint):
    data_dir = args.data_dir
    skip = args.skip
    num_random_sample = args.num_random_sample
    # ../data/random_test_0.8_100
    if os.path.exists(
            os.path.join(data_dir, 'random_test_' + str(similarity_threshold) + '_' + str(perturbation_constraint))):
        return
    if args.task_name == 'snli':
        _randomize_snli_testset(data_dir, skip, similarity_threshold, perturbation_constraint, num_random_sample, random_smooth)


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

    counterfitted_tv_file = os.path.join(cache_dir, task_name + '_counterfitted_tv_pca' + str(similarity_threshold) + '_' + str(
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
    elif args.model_type == 'bow':
        model = BOWModel(new_state_dict['embedding.weight'], n_vocab=args.vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size, num_classes=args.num_labels)
        model.load_state_dict(new_state_dict)
    elif args.model_type == 'decom_att':
        model = DecompAttentionModel(args.word_mat, n_vocab=args.vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size, num_classes=args.num_labels)
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


def _get_tv(text, counterfitted_tv, key_set, exclude):
    tokens = text.split(' ')
    tv_list = np.zeros(len(tokens))
    for i, token in enumerate(tokens):
        if len(token) > 0 and token[-1] in exclude:
            token = token[:-1]

        if token in key_set:
            tv_list[i] = counterfitted_tv[token]
        else:
            tv_list[i] = 1.
    return np.sort(tv_list)


def _load_and_cache_bert_example(args, tokenizer, folder, type='test'):
    '''
    load or cache the InputExample, return dataset
    :param type:
    :param args:
    :param task_class:
    :param tokenizer:
    :return:
    '''
    task_class = processors[args.task_name]()
    file = 'normal_bert_cached_{}_{}_{}'.format(
        type,
        args.max_seq_length,
        args.task_name
    )
    cached_features_file = os.path.join(folder, file)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
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


def _load_and_cache_normal_example(args, word2index, folder, type='test'):
    task_class = processors[args.task_name]()
    # file: cached_train_bert-base-uncased_256_imdb
    file = 'normal_cached_{}_{}_{}'.format(
        type,
        args.max_seq_length,
        args.task_name
    )
    cached_features_file = os.path.join(folder, file)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        pad_token = 0
        examples = torch.load(folder + '/example')
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


def _certify_randomized_smooth(args, random_data_dir, tokenizer, word2index, model, counterfitted_tv, key_set):
    '''
    :param random_data_dir: ../data/random_test_0.8_100/
    :return:
    '''
    similarity_threshold = args.similarity_threshold
    task_name = args.task_name

    text_count = 0
    total_cert_count = 0
    total_count = 0

    if os.path.exists(random_data_dir):
        files = os.listdir(random_data_dir)  # dirs
        print('tqdm evaluate each text ...')
        for file in tqdm.tqdm(files):  # file is a dir
            # read original text
            original_text_file = os.path.join(args.data_dir, 'test', file + '.txt')
            original_text = open(original_text_file).read()
            if args.model_type == 'bert':
                eval_dataset = _load_and_cache_bert_example(args, tokenizer, os.path.join(random_data_dir, file))
            else:
                eval_dataset = _load_and_cache_normal_example(args, word2index, os.path.join(random_data_dir, file))
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
                label_ids = batch[4].detach().cpu().numpy()
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

    return total_count, total_cert_count


def _randomized_evaluate_imdb(args, tokenizer, word2index, model, counterfitted_tv, key_set):
    random_data_dir = os.path.join(args.data_dir, 'random_test_' + str(args.similarity_threshold) + '_' + str(args.perturbation_constraint))
    total_count1, total_cert_count1 = _certify_randomized_smooth(args, random_data_dir, tokenizer, word2index, model, counterfitted_tv, key_set)
    print('certify acc is {:.4f}'.format(total_cert_count1 / total_count1))


def randomized_evaluate(args, model, tokenizer, word2index):
    # read tv perturb
    similarity_threshold = args.similarity_threshold
    cache_dir = args.cache_dir
    task_name = args.task_name

    counterfitted_tv_file = os.path.join(cache_dir, task_name + '_counterfitted_tv_pca' + str(similarity_threshold) + '_' + str(
                                             args.perturbation_constraint) + '.pkl')
    with open(counterfitted_tv_file, 'rb') as f:
        counterfitted_tv = pickle.load(f)  # dict
        key_set = set(counterfitted_tv.keys())

    if task_name == 'snli':
        _randomized_evaluate_imdb(args, tokenizer, word2index, model, counterfitted_tv, key_set)



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
    parser.add_argument("--max_seq_length", default=128, type=int,
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
    # load vocab.
    word2index = None
    if args.model_type != 'bert':
        with open(args.cache_dir + '/{}_vocab_train.pkl'.format(args.task_name), 'rb') as f:
            vocab = pickle.load(f)
        index2word = vocab['index2word']
        word2index = vocab['word2index']
        word_mat = vocab['word_mat']
        args.word_mat = word_mat
        args.vocab_size = len(index2word)

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
    model.to(device)

    similarity_threshold = args.similarity_threshold
    perturbation_constraint = args.perturbation_constraint

    perturbation_file = args.cache_dir + '/' + args.task_name + '_perturbation_constraint_pca' + str(
        similarity_threshold) + "_" + str(perturbation_constraint) + '.pkl'
    with open(perturbation_file, 'rb') as f:
        perturb = pickle.load(f)

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
    randomized_evaluate(args, model, tokenizer, word2index)


if __name__ == '__main__':
    main()

