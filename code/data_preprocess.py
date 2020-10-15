# -*- coding: utf-8 -*-#
# Name:         data_preprocess
# Description:  
# Author:       Fali Wang
# Date:         2020/8/28 21:49
import argparse
import os
import numpy as np
import pickle
import string
import pandas as pd
import tqdm
from nltk import word_tokenize
import networkx as nx
import json


def get_word_embed(counter_fitted_path, embed_file):
    '''
    read counter-fitting embedding file, save to embed_file.
    :param counter_fitted_path:
    :param embed_file:
    :return:
    '''
    if os.path.exists(embed_file):
        with open(embed_file, 'rb') as input_file:
            word_embed = pickle.load(input_file)
    else:
        print('Generate Word Embedding.')
        word_embed = {}
        file = os.path.join(counter_fitted_path, 'counter-fitted-vectors.txt')
        with open(file, "r") as f:
            tem = f.readlines()
            for line in tem:
                line = line.strip()
                line = line.split(' ')
                word = line[0]
                vec = line[1:]
                vec = [float(i) for i in vec]
                vec = np.array(vec)
                word_embed[word] = vec

        with open(embed_file, 'wb') as output:
            pickle.dump(word_embed, output)
    return word_embed


def _get_imdb_vocab_by_folder(folder_lists, word_embed):
    vocab = {}
    for folder_list in folder_lists:
        for folder in os.listdir(folder_list):
            sub_folder = os.path.join(folder_list, folder)
            if os.path.isdir(sub_folder):
                for input_file in os.listdir(sub_folder):
                    file = os.path.join(sub_folder, input_file)
                    with open(file, 'r', encoding='utf-8') as f:
                        tem_text = f.readlines()
                        if tem_text:
                            tem_text = tem_text[0].translate(str.maketrans('', '', string.punctuation))
                            for word in tem_text.split(' '):
                                if word in vocab:
                                    vocab[word]['freq'] += 1
                                elif word in word_embed:
                                    vocab[word] = {'freq': 1, 'vec': word_embed[word]}
    return vocab


def _get_amazon_vocab_by_csv(vocab, file, word_embed):
    df = pd.read_csv(file, header=None)
    n_train = df.shape[0]
    print('Amazon create vocabulary tqdm...')
    for i in tqdm.tqdm(range(n_train)):
        x_raw = df.loc[i, 2]  # colum 2 is review
        x_tokens = word_tokenize(x_raw)
        for word in x_tokens:
            if word in vocab:
                vocab[word]['freq'] += 1
            elif word in word_embed:
                vocab[word] = {'freq': 1, 'vec': word_embed[word]}


def get_vocabulary(task_name, data_path, word_embed_dict_or_file , vocab_file):
    '''
    the 交集 between Counter-fitting and vocab
    :param task_name:
    :param data_path:
    :param word_embed_dict_or_file:
    :param vocab_file:
    :return:
    '''
    if os.path.exists(vocab_file):
        print("Loading vocab from {}.".format(vocab_file))
        with open(vocab_file, 'rb') as input_file:
            vocab = pickle.load(input_file)
    else:
        if isinstance(word_embed_dict_or_file, dict):
            word_embed = word_embed_dict_or_file
        else:
            with open(word_embed_dict_or_file, 'rb') as input_file:
                word_embed = pickle.load(input_file)

        print('Generate vocabulary.')
        if task_name == 'imdb':
            folder_lists = [data_path + '/test', data_path + '/train']
            vocab = _get_imdb_vocab_by_folder(folder_lists, word_embed)
            with open(vocab_file, 'wb') as output:
                pickle.dump(vocab, output)
            print('Finish generate Imdb vocabulary.')
        elif task_name == 'amazon':
            vocab = {}
            _get_amazon_vocab_by_csv(vocab, data_path + '/train.csv', word_embed)
            _get_amazon_vocab_by_csv(vocab, data_path + '/test.csv', word_embed)
            with open(vocab_file, 'wb') as output:
                pickle.dump(vocab, output)
            print('Finish generate Amazon vocabulary.')
        elif task_name == 'snli':
            vocab = {}
            file_lists = [data_path + '/snli_1.0_train.jsonl', data_path + '/snli_1.0_test.jsonl']
            for file in file_lists:
                with open(file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        prem, hypo = data['sentence1'].lower(), data['sentence2'].lower()
                        prem = prem.translate(str.maketrans('', '', string.punctuation))
                        hypo = hypo.translate(str.maketrans('', '', string.punctuation))
                        tokens = prem.split() + hypo.split()
                        for word in tokens:
                            if word in vocab:
                                vocab[word]['freq'] += 1
                            elif word in word_embed:
                                vocab[word] = {'freq': 1, 'vec': word_embed[word]}
            with open(vocab_file, 'wb') as output:
                pickle.dump(vocab, output)
            print('Finish generate SNLI vocabulary.')
    return vocab


def process_with_all_but_not_top(task_name, cache_path):
    # code for processing word embd using all-but-not-top
    embed_pca_file = cache_path + "/{}_embed_pca.pkl".format(task_name)
    vocab_pca_file = cache_path + "/{}_vocab_pca.pkl".format(task_name)
    if os.path.exists(embed_pca_file) and os.path.exists(vocab_pca_file):
        return

    print('Process word embd using all-but-not-top')
    vocab_file = cache_path + "/{}_vocab.pkl".format(task_name)
    pkl_file = open(vocab_file, 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()

    num_word = len(vocab)
    dim_vec = len(vocab['high']['vec'])
    embd_matrix = np.zeros([num_word, dim_vec])  # 向量矩阵
    embd_matrix0 = np.zeros([num_word, dim_vec])

    count = 0
    tem_list = []  # 单词列表
    for key in vocab.keys():
        tem_vec = vocab[key]['vec']
        tem_vec = tem_vec / np.sqrt((tem_vec ** 2).sum())
        embd_matrix[count, :] = tem_vec
        tem_list.append(key)
        count += 1

    mean_embd_matrix = np.mean(embd_matrix, axis=0)
    for i in range(embd_matrix.shape[0]):
        embd_matrix0[i, :] = embd_matrix[i, :] - mean_embd_matrix
    covMat = np.cov(embd_matrix0, rowvar=0)  # 计算协方差，以列为单位
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(-eigVals)
    eigValIndice = eigValIndice[0:8]
    n_eigVect = eigVects[:, eigValIndice]
    embd_matrix = embd_matrix0 - np.dot(np.dot(embd_matrix, n_eigVect), n_eigVect.T)  # type is mat

    output = open(embed_pca_file, 'wb')
    pickle.dump(embd_matrix, output)
    output.close()

    # update vocabulary
    count = 0
    for key in tem_list:
        vocab[key]['vec'] = embd_matrix[count, :].flatten()
        count += 1

    output = open(vocab_pca_file, 'wb')
    pickle.dump(vocab, output)
    output.close()

    print('Finish Process word embd using all-but-not-top')


def get_word_substitution_table(task_name, cache_path, similarity_threshold, neighbor_file):
    # neighbor_file = cache_path + '/{}_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'
    if os.path.exists(neighbor_file):
        return
    print("Generate word substitude table.")
    vocab_file = cache_path + '/{}_vocab_pca.pkl'.format(task_name)
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    counterfitted_neighbor = {}
    key_list = list(vocab.keys())
    similarity_num_threshold = 100000
    freq_threshold = 1
    neighbor_network_node_list = []
    neighbor_netword_link_list = []

    num_word = len(vocab)
    dim_vec = vocab[key_list[0]]['vec'].shape[1]

    embed_matrix = np.zeros([num_word, dim_vec])
    for i in range(num_word):
        embed_matrix[i, :] = vocab[key_list[i]]['vec']

    print('get word substitution table... 42171')
    for i in tqdm.tqdm(range(num_word)):
        word = key_list[i]
        if vocab[word]['freq'] > freq_threshold:
            counterfitted_neighbor[word] = []
            neighbor_network_node_list.append(word)

            dist_vec = np.dot(embed_matrix[i, :], embed_matrix.T)
            dist_vec = np.array(dist_vec).flatten()  # similarity

            # idxes = np.argsort(-dist_vec)
            idxes = np.where(dist_vec > similarity_threshold)  # return coordinate
            idxes = idxes[0].tolist()

            tem_num_count = 0
            for ids in idxes:
                if key_list[ids] != word and vocab[key_list[ids]]['freq'] > freq_threshold:
                    counterfitted_neighbor[word].append(key_list[ids])
                    neighbor_netword_link_list.append((word, key_list[ids]))
                    tem_num_count += 1
                    if tem_num_count >= similarity_num_threshold:
                        break
    neighbor = {'neighbor': counterfitted_neighbor, 'link': neighbor_netword_link_list, 'node': neighbor_network_node_list}
    with open(neighbor_file, 'wb') as f:
        pickle.dump(neighbor, f)
    print('Finish generate word substitude table.')


def get_perturbation_set(task_name, cache_path, similarity_threshold, perturbation_constraint, perturbation_file):
    if os.path.exists(perturbation_file):
        return
    print('Generate perturbation set.')
    with open(cache_path + '/{}_vocab_pca.pkl'.format(task_name), 'rb') as f:
        vocab = pickle.load(f)
    with open(cache_path + '/{}_neighbor_constraint_pca{}.pkl'.format(task_name, similarity_threshold), 'rb') as f:
        neighbor = pickle.load(f)

    freq_threshold = 1
    conterfitted_neighbor = neighbor['neighbor']
    neighbor_network_node_list = neighbor['node']
    neighbor_network_link_list = neighbor['link']
    perturb = {}

    key_list = list(vocab.keys())
    num_word = len(key_list)
    dim_vec = vocab[key_list[0]]['vec'].shape[1]
    embed_matrix = np.zeros([num_word, dim_vec])
    for i in range(num_word):
        embed_matrix[i, :] = vocab[key_list[i]]['vec']

    # find connected subgraph
    G = nx.Graph()
    for node in neighbor_network_node_list:
        G.add_node(node)
    for link in neighbor_network_link_list:
        G.add_edge(link[0], link[1])

    for c in nx.connected_components(G):  # set
        node_set = G.subgraph(c).nodes()
        if len(node_set) <= 1:
            continue
        elif len(node_set) <= perturbation_constraint:
            tem_key_list = list(node_set)
            tem_num_word = len(tem_key_list)
            tem_embed_matrix = np.zeros([tem_num_word, dim_vec])
            for i in range(tem_num_word):
                tem_embed_matrix[i, :] = vocab[tem_key_list[i]]['vec']
            for node in node_set:
                perturb[node] = {'set': [], 'isdivide': 0}
                dist_vec = np.dot(vocab[node]['vec'], tem_embed_matrix.T)
                dist_vec = np.array(dist_vec).flatten()
                idxes = np.argsort(-dist_vec)
                tem_list = []
                for ids in idxes:
                    if vocab[tem_key_list[ids]]['freq'] > freq_threshold:
                        tem_list.append(tem_key_list[ids])
                perturb[node]['set'] = tem_list
        else:
            tem_key_list = list(node_set)
            tem_num_word = len(tem_key_list)
            tem_embed_matrix = np.zeros([tem_num_word, dim_vec])
            for i in range(tem_num_word):
                tem_embed_matrix[i, :] = vocab[tem_key_list[i]]['vec']
            for node in node_set:
                perturb[node] = {'set': [], 'isdivide': 1}
                dist_vec = np.dot(vocab[node]['vec'], tem_embed_matrix.T)
                dist_vec = np.array(dist_vec).flatten()
                idxes = np.argsort(-dist_vec)
                tem_list = []
                for j, ids in enumerate(idxes):
                    if vocab[tem_key_list[ids]]['freq'] > freq_threshold:
                        tem_list.append(tem_key_list[ids])
                    if j+1 == perturbation_constraint:
                        break
                perturb[node]['set'] = tem_list
    with open(perturbation_file, 'wb') as f:
        pickle.dump(perturb, f)
    print('Generate perturbation set finished.')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of task is selected in [imdb, amazon]")
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir.")
    parser.add_argument("--cache_dir", default='../cache', type=str, help="The cache data dir.")
    parser.add_argument("--embed_dir", default=None, type=str, required=True, help="Counter-fitting embedding dir.")
    parser.add_argument("--similarity_threshold", default=0.8, type=float, help="Similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", "-K", default=100, type=int, help="Maximum size of perturbation set of each word.")

    args = parser.parse_args()

    data_path = args.data_dir
    cache_path = args.cache_dir
    counter_fitted_path = args.embed_dir
    task_name = args.task_name
    similarity_threshold = args.similarity_threshold
    perturbation_constraint = args.perturbation_constraint

    # use pickle dump 1.word_embed is Counter-fitting Glove word vector's improvement, dict {'apple': [0.1,...,0.0],
    # } 2.{task_name}_vocab.pkl, imdb remove punctuation then split(' '), amazon tokenizer use nltk,
    # not bert tokenizer, dict {'apple':{vec:[], 'freq':1},...} 42171 words 3.process word embedding using
    # all-but-not-top, update vocab mainly, 'All-but-the-Top: Simple and Effective Postprocessing for Word
    # Representations' 4. create word substitution table, include 0.91 neighbors/w without itself, 37086 nodes,
    # 33596 links 5. get pertubation set, dict {'apple':{'set':[], 'isdivide':0/1}}, 9.1 pertubation/w with itself,
    # 11729 words

    # 1.
    embed_file = os.path.join(cache_path, 'word_embed.pkl')
    word_embed_dict = get_word_embed(counter_fitted_path, embed_file)
    # 2.
    vocab_file = os.path.join(cache_path, '{}_vocab.pkl'.format(task_name))
    get_vocabulary(task_name, data_path, word_embed_dict, vocab_file)
    # 3.
    process_with_all_but_not_top(task_name, cache_path)
    # 4.
    neighbor_file = cache_path + '/' + task_name + '_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'
    get_word_substitution_table(task_name, cache_path, similarity_threshold, neighbor_file)
    # 5. generate perturbation set
    perturbation_file = cache_path + '/' + task_name + '_perturbation_constraint_pca' + str(similarity_threshold) + "_" + str(perturbation_constraint) + '.pkl'
    get_perturbation_set(task_name, cache_path, similarity_threshold, perturbation_constraint, perturbation_file)


if __name__ == '__main__':
    main()