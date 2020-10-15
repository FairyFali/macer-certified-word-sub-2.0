# -*- coding: utf-8 -*-#
# Name:         snli_dataset
# Description:  
# Author:       Fali Wang
# Date:         2020/10/9 11:50
import os
import tqdm
import json
from enum import Enum

from utils import DataProcessor, InputExample


class EntailmentLabels(Enum):
    contradiction = 0
    neutral = 1
    entailment = 2


def _get_snli_examples(data_dir, totals, type):
    examples = []
    fn = os.path.join(data_dir, 'snli_1.0_{}.jsonl'.format(type))
    with open(fn) as f:
        for line in tqdm.tqdm(f, total=totals[type]):
            data = json.loads(line)
            prem, hypo, gold_label = data['sentence1'], data['sentence2'], data['gold_label']
            try:
                gold_label = EntailmentLabels[gold_label].value
            except KeyError:
                # Encountered gold label '-', can't use so skip it
                continue
            example = InputExample(guid=None, text_a=prem, text_b=hypo, label=gold_label)
            examples.append(example)
    return examples


class SNLIProcessor(DataProcessor):
    totals = {'train': 550152, 'dev': 10000, 'test': 10000}

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return _get_snli_examples(data_dir, self.totals, 'train')

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return _get_snli_examples(data_dir, self.totals, 'dev')

    def get_test_examples(self, data_dir):
        return _get_snli_examples(data_dir, self.totals, 'test')

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [0, 1, 2]


