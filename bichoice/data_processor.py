import abc
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



class C3BinaryExample(object):
    '''
    a single bi-choice example in C3 dataset
    '''

    def __init__(self, passage, question, choice_0, choice_1, label=None, **kws):
        self.passage = passage
        self.question = question
        self.choice_0 = choice_0
        self.choice_1 = choice_1
        self.label = label


    def __str__(self):
        return str(dict({
            'passage': self.passage,
            'question': self.question,
            'choice_0': self.choice_0,
            'choice_1': self.choice_1,
            'label': self.label if self.label is not None else 'NA'
        }))



class DatasetGetter(abc.ABC):

    @abc.abstractmethod
    def get_dataset(self):
        '''return a `torch.utils.data.Dataset` object'''
        raise NotImplementedError()

    @abc.abstractmethod
    def convert_example_to_features(self):
        '''return a `InputFeatures` object'''
        raise NotImplementedError()



class C3BinaryDataProcessor(DatasetGetter):
    '''
    return C3 dataset as a binary classification problem
    via an implemented method `get_dataset`.
    '''
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length


    def get_dataset(self, fn, with_label=True):
        '''
        return C3 dataset as a binary classification problem.

        Args
        ----
        `fn` : path to csv file of the dataset

        `with_label` : whether that dataset is labeled.

        Return
        ------
        `out` : `TensorDataset` object.
        '''
        features = []

        df = pd.read_csv(fn)
        for i in tqdm(df.index, desc='tokenizing'):
            example = df.iloc[i]
            features.append(self.convert_example_to_features(example))

        all_input_ids = torch.cat([
            torch.LongTensor([f.input_ids]) for f in features
        ])
        all_input_mask = torch.cat([
            torch.LongTensor([f.input_mask]) for f in features
        ])
        all_segment_ids = torch.cat([
            torch.LongTensor([f.segment_ids]) for f in features
        ])
        
        if with_label:
            all_label_ids = torch.LongTensor(df.label.values)
            return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        else:
            return TensorDataset(all_input_ids, all_input_mask, all_segment_ids)


    def convert_example_to_features(self, example):
        '''
        convert a single example to a single `InputFeatures` object.

        [CLS] passage [SEP] question [SEP] choice_0 [SEP] chioce_1 [SEP]

        Args
        ----
        `example` : instance of `C3BinaryExample` or any other object that has same attributes.

        Return
        ------
        `out` : `InputFeatures` object
        '''
        tokens_a = self.tokenizer.tokenize(example.passage)
        tokens_b = self.tokenizer.tokenize(example.question)
        tokens_c = self.tokenizer.tokenize(example.choice_0)
        tokens_d = self.tokenizer.tokenize(example.choice_1)

        # truncate to max length - 5 (there are 5 special tokens)
        self._truncate_seq_tuple(tokens_a, tokens_b, tokens_c, tokens_d, max_length=self.max_length-5)
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]'] + tokens_c + ['[SEP]'] + tokens_d + ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0]*(2+len(tokens_a)) + [1]*(2+len(tokens_b)+len(tokens_c))
        # zeor padding
        input_ids += [0] * (self.max_length-len(input_ids))
        input_mask += [0] * (self.max_length-len(input_mask))
        segment_ids += [0] * (self.max_length-len(segment_ids))

        assert len(input_ids) == self.max_length
        assert len(input_mask) == self.max_length
        assert len(segment_ids) == self.max_length

        label = None
        if hasattr(example, 'label'):
            label = example.label

        return InputFeatures(input_ids, input_mask, segment_ids, label)

    def _truncate_seq_tuple(self, *tokenList, max_length):
        '''
        Truncates a sequence tuple in place to the maximum length.

        This is a simple heuristic which will always truncate the
        longer sequence one token at a time. This makes more sense
        than truncating an equal percent of tokens from each, since if
        one sequence is very short then each token that's truncated
        likely contains more information than a longer sequence.
        '''
        while True:
            lengthList = [len(a) for a in tokenList]
            if sum(lengthList) <= max_length:
                break
            # find longest tokens
            i = np.argmax(lengthList)
            # then truncate it
            tokenList[i].pop()
