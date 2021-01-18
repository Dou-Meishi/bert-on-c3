
import json
import random

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

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(self.dataset[0], "train")

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self.dataset[1], "dev")

    def get_test_examples(self):
        return self._create_examples(self.dataset[2], "test")

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

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        """ 各样本格式：CLS document SEP question SEP choice SEP
        """

        print("#examples", len(examples))

        label_map = {}
        for (i, label) in enumerate(self.get_labels()):
            label_map[label] = i

        features = [[]]
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = tokenizer.tokenize(example.text_b)

            tokens_c = tokenizer.tokenize(example.text_c)

            self._truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
            tokens_b = tokens_c + ["[SEP]"] + tokens_b

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]

            features[-1].append(
                    InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id))
            if len(features[-1]) == 4:
                features.append([])

        if len(features[-1]) == 0:
            features = features[:-1]
        print('#features', len(features))
        return features

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

    def get_dataset(self, examples, tokenizer, max_length, is_test=False, use_cache=True):
        """ 每一个样本有4个子样本组成，子样本是各个选项和问题、上下文的拼接，子样本间共享label
                CLS doc SEP question SEP choice-A SEP
                CLS doc SEP question SEP choice-B SEP
                CLS doc SEP question SEP choice-C SEP
                CLS doc SEP question SEP choice-D SEP
            因此，维度是 batch size * 4 * seq length
        """
        features = self.convert_examples_to_features(examples, max_length, tokenizer)
        input_ids, input_mask, segment_ids = [], [], []

        label_id = []
        for f in features:
            # new_ids, new_mask, new_segment_ids = [], [], []
            input_ids.append([])
            input_mask.append([])
            segment_ids.append([])
            for i in range(len(self.get_labels())):
                input_ids[-1].append(f[i].input_ids)
                input_mask[-1].append(f[i].input_mask)
                segment_ids[-1].append(f[i].segment_ids)
                # new_ids.append(f[i].input_ids)
                # new_mask.append(f[i].input_mask)
                # new_segment_ids.append(f[i].segment_ids)

            # input_ids.append(new_ids)
            # input_mask.append(new_mask)
            # segment_ids.append(new_segment_ids)
            if not is_test:
                label_id.append([f[0].label_id])

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        if not is_test:
            all_label_ids = torch.tensor(label_id, dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        return dataset

