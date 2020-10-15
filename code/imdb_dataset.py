# -*- coding: utf-8 -*-#
# Name:         imdb_dataset
# Description:  
# Author:       Fali Wang
# Date:         2020/8/26 17:10
import os

from utils import DataProcessor, InputExample


def _create_examples_from_folder(folder_list, set_type):
    '''
    create examples for train and dev sets
    :param folder_list:
    :param set_type: train or dev or test
    :return:
    '''
    examples = []
    i = 0
    for folder in folder_list:
        label_dir = os.path.basename(folder)  # last dir name
        for input_file in os.listdir(folder):
            if input_file.endswith('.txt'):
                with open(os.path.join(folder, input_file), 'r', encoding='utf-8') as f:
                    text = f.readlines()
                    if text:
                        guid = "%s-%d" % (set_type, i); i += 1
                        text_a = text[0]
                        text_b = None
                        label = 1 if label_dir == 'pos' else 0
                        examples.append(
                            InputExample(guid, text_a, text_b, label)
                        )
    return examples


class ImdbProcessor(DataProcessor):
    '''
    IMDB Movie Review data set.
    '''
    def get_train_examples(self, data_dir):
        '''
        Gets a collection of `InputExample`s for the train set.
        :param data_dir: IMDB Data dir, ../data/aclImdb
        :return:
        '''
        folder1 = os.path.join(data_dir, "train", "pos")
        folder2 = os.path.join(data_dir, "train", "neg")
        return _create_examples_from_folder([folder1, folder2], "train")

    def get_dev_examples(self, data_dir):
        '''
        Similar with above.
        :param data_dir:
        :return:
        '''
        folder1 = os.path.join(data_dir, "test", "pos")
        folder2 = os.path.join(data_dir, "test", "neg")
        return _create_examples_from_folder([folder1, folder2], "dev")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [0, 1]
