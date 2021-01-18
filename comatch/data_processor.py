
import os
import json
import random
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b, label_id):
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.segment_ids_a = segment_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.segment_ids_b = segment_ids_b
        self.label_id = label_id

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
   

class DataProcessor(object):
    def __init__(self, data_dir):
        """ 原来的样本：1篇文章，1个问题，4个选项，1个标签
            现在样本：1篇文章，1个问题，1个选项，1个标签，将原来1个样本拆分为4个样本
        """
        random.seed(42)
        self.dataset = [[], [], []]

        # 文章，问题，4个选项（不足4个，补充‘’），答案

        for sid in range(3):
            data = []
            for subtask in ["d", "m"]:
                with open(data_dir + "c3-" + subtask + "-" + ["train.json", "dev.json", "test.json"][sid], "r",
                          encoding="utf8") as f:
                    data += json.load(f)
            if sid == 0:
               random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
                    for k in range(len(data[i][1][j]["choice"])):
                        d += [data[i][1][j]["choice"][k].lower()]
                    for k in range(len(data[i][1][j]["choice"]), 4):
                        d += ['']
                    d += [data[i][1][j]["answer"].lower()]
                    self.dataset[sid] += [d]

    def get_examples(self, subset):
        if subset == 'train':
            return self._create_examples(self.dataset[0], "train")
        if subset == 'valid':
            return self._create_examples(self.dataset[1], "dev")
        if subset == 'test':
            return self._create_examples(self.dataset[2], "test")

    # def get_train_examples(self):
    #     """Gets a collection of `InputExample`s for the train set."""
    #     return self._create_examples(self.dataset[0], "train")
    #
    # def get_dev_examples(self):
    #     """Gets a collection of `InputExample`s for the dev set."""
    #     return self._create_examples(self.dataset[1], "dev")
    #
    # def get_test_examples(self):
    #     return self._create_examples(self.dataset[2], "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        answer = -1
        examples = []
        for (i, d) in enumerate(data):
            for k in range(4):
                if data[i][2 + k] == data[i][6]:
                    answer = str(k)
            label = convert_to_unicode(answer)

            for k in range(4):
                guid = "%s-%s-%s" % (set_type, i, k)
                text_a = convert_to_unicode(data[i][0])
                text_b = convert_to_unicode(data[i][k + 2])
                text_c = convert_to_unicode(data[i][1])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))

        return examples

    def convert_examples_to_features(self, examples, doc_max_seq_length, ques_max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        """ 各样本格式：CLS document SEP question SEP choice SEP
        """

        print("#examples", len(examples))

        label_map = {}
        for (i, label) in enumerate(self.get_labels()):
            label_map[label] = i

        features = [[]]
        for (ex_index, example) in enumerate(tqdm(examples)):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = tokenizer.tokenize(example.text_b)

            tokens_c = tokenizer.tokenize(example.text_c)

            self._truncate_seq(tokens_a, doc_max_seq_length - 2)
            self._truncate_seq_pair(tokens_b, tokens_c, ques_max_seq_length - 3)

            # self._truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
            tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
            tokens_b = ["[CLS]"] + tokens_c + ["[SEP]"] + tokens_b + ["[SEP]"]

            segment_ids_a = [0] * len(tokens_a)
            segment_ids_b = [0] * (len(tokens_c) + 2) + [1] * (len(tokens_b) - len(tokens_c) - 2)

            input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
            input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)


            # tokens = []
            # segment_ids = []
            # tokens.append("[CLS]")
            # segment_ids.append(0)
            # for token in tokens_a:
            #     tokens.append(token)
            #     segment_ids.append(0)
            # tokens.append("[SEP]")
            # segment_ids.append(0)

            # if tokens_b:
            #     for token in tokens_b:
            #         tokens.append(token)
            #         segment_ids.append(1)
            #     tokens.append("[SEP]")
            #     segment_ids.append(1)

            # input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask_a = [1] * len(input_ids_a)
            input_mask_b = [1] * len(input_ids_b)
            # Zero-pad up to the sequence length.
            if len(input_ids_a) < doc_max_seq_length:
                shorten_length = doc_max_seq_length - len(input_ids_a)
                input_ids_a = input_ids_a + [0] * shorten_length
                input_mask_a = input_mask_a + [0] * shorten_length
                segment_ids_a = segment_ids_a + [0] * shorten_length

            if len(input_ids_b) < ques_max_seq_length:
                shorten_length = ques_max_seq_length - len(input_ids_b)
                input_ids_b = input_ids_b + [0] * shorten_length
                input_mask_b = input_mask_b + [0] * shorten_length
                segment_ids_b = segment_ids_b + [0] * shorten_length

            #assert len(input_ids) == max_seq_length
            #assert len(input_mask) == max_seq_length
            #assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]

            features[-1].append(
                    InputFeatures(
                            input_ids_a=input_ids_a,
                            input_mask_a=input_mask_a,
                            segment_ids_a=segment_ids_a,
                            input_ids_b=input_ids_b,
                            input_mask_b=input_mask_b,
                            segment_ids_b=segment_ids_b,
                            label_id=label_id))
            if len(features[-1]) == 4:
                features.append([])

        if len(features[-1]) == 0:
            features = features[:-1]
        print('#features', len(features))
        return features

    def _truncate_seq(self, tokens, max_length):
        while True:
            if len(tokens) <= max_length:
                break
            tokens.pop()

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _truncate_seq_tuple(self, tokens_a, tokens_b, tokens_c, max_length):
        """Truncates a sequence tuple in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
                tokens_b.pop()
            else:
                tokens_c.pop()

    def get_dataset(self, tokenizer, doc_max_length=512, ques_max_length=128, subset='train', use_cache=True):
        """ 每一个样本有4个子样本组成，子样本是各个选项和问题、上下文的拼接，子样本间共享label
                CLS doc SEP question SEP choice-A SEP
                CLS doc SEP question SEP choice-B SEP
                CLS doc SEP question SEP choice-C SEP
                CLS doc SEP question SEP choice-D SEP
            因此，维度是 batch size * 4 * seq length
        """
        cache_file = './cached_files/' + subset + '_dataset.pt'
        if use_cache and os.path.exists(cache_file):
            dataset = torch.load(cache_file)
            return dataset

        examples = self.get_examples(subset)

        features = self.convert_examples_to_features(examples, doc_max_length, ques_max_length, tokenizer)
        input_ids_a, input_mask_a, segment_ids_a = [], [], []
        input_ids_b, input_mask_b, segment_ids_b = [], [], []
        label_id = []
        for f in tqdm(features):
            # new_ids, new_mask, new_segment_ids = [], [], []
            input_ids_b.append([])
            input_mask_b.append([])
            segment_ids_b.append([])
            for i in range(len(self.get_labels())):
                # tokenb : ques choice
                input_ids_b[-1].append(f[i].input_ids_b)
                input_mask_b[-1].append(f[i].input_mask_b)
                segment_ids_b[-1].append(f[i].segment_ids_b)

            # doc input ids mask segement
            input_ids_a.append(f[0].input_ids_a)
            input_mask_a.append(f[0].input_mask_a)
            segment_ids_a.append(f[0].segment_ids_a)
            # if subset != 'test':
            label_id.append([f[0].label_id])

        all_input_ids_a = torch.tensor(input_ids_a, dtype=torch.long)
        all_input_mask_a = torch.tensor(input_mask_a, dtype=torch.long)
        all_segment_ids_a = torch.tensor(segment_ids_a, dtype=torch.long)
        all_input_ids_b = torch.tensor(input_ids_b, dtype=torch.long)
        all_input_mask_b = torch.tensor(input_mask_b, dtype=torch.long)
        all_segment_ids_b = torch.tensor(segment_ids_b, dtype=torch.long)
        # if subset != 'test':
        all_label_ids = torch.tensor(label_id, dtype=torch.long)
        dataset = TensorDataset(all_input_ids_a, all_input_mask_a, all_segment_ids_a, all_input_ids_b, all_input_mask_b, all_segment_ids_b, all_label_ids)
        # else:
        #     dataset = TensorDataset(all_input_ids_a, all_input_mask_a, all_segment_ids_a,all_input_ids_b, all_input_mask_b, all_segment_ids_b)

        if use_cache:
            if not os.path.exists('./cached_files/'):
                os.mkdir('./cached_files/')
            torch.save(dataset, cache_file)

        return dataset

