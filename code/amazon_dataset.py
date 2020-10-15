# -*- coding: utf-8 -*-#
# Name:         amazon_dataset
# Description:  
# Author:       Fali Wang
# Date:         2020/8/26 21:08

import pandas as pd
import os
import tqdm

from utils import DataProcessor, InputExample


def _generate_example(data_dir, sub_dir):
    # text len 80% is 116, 85% is 127
    data = pd.read_csv(os.path.join(data_dir, sub_dir + '.csv'), header=None)
    examples = []

    n = data.shape[0]
    for i in tqdm.tqdm(range(n)):
        # label is [0,1,2,3,4]
        label = data.loc[i, 0] - 1
        text_a = data.loc[i, 2]
        guid = "%s-%d" % (sub_dir, i)
        text_b = None
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples


class AmazonProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return _generate_example(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return _generate_example(data_dir, 'test')

    def get_labels(self):
        return [0, 1, 2, 3, 4]

