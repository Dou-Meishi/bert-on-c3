
import os
import json
import random
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, context, question, option_0, option_1, option_2, option_3, label=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.context = context
        self.question = question
        self.options = [
            option_0,
            option_1,
            option_2,
            option_3,
        ]
        self.label = label

class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.label = label


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

class DCMNDataProcessor(object):
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

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        answer = -1
        examples = []
        for (i, d) in enumerate(tqdm(data)):
            for k in range(4):
                if data[i][2 + k] == data[i][6]:
                    answer = k

            examples.append(InputExample(
                guid="%s-%s" % (set_type, i),
                context=convert_to_unicode(data[i][0]),
                question=convert_to_unicode(data[i][1]),
                option_0=convert_to_unicode(data[i][2]),
                option_1=convert_to_unicode(data[i][3]),
                option_2=convert_to_unicode(data[i][4]),
                option_3=convert_to_unicode(data[i][5]),
                label=answer
            ))

        return examples

    def get_examples(self, subset):
        if subset == 'train':
            return self._create_examples(self.dataset[0], "train")
        if subset == 'valid':
            return self._create_examples(self.dataset[1], "dev")
        if subset == 'test':
            return self._create_examples(self.dataset[2], "test")

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        pop_label = True
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(1)
            else:
                tokens_b.pop(1)

    def convert_examples_to_features(self, examples, tokenizer, max_seq_length, is_training=True):
        """Loads a data file into a list of `InputBatch`s."""

        # Swag is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        features = []
        for example_index, example in enumerate(tqdm(examples)):
            context_tokens = tokenizer.tokenize(example.context)
            question_tokens = tokenizer.tokenize(example.question)

            choices_features = []
            for option_index, option in enumerate(example.options):
                # We create a copy of the context tokens in order to be
                # able to shrink it according to option_tokens
                context_tokens_choice = context_tokens[:]  # + question_tokens

                option_token = tokenizer.tokenize(option)
                option_len = len(option_token)
                ques_len = len(question_tokens)

                option_tokens = question_tokens + option_token

                # Modifies `context_tokens_choice` and `option_tokens` in
                # place so that the total length is less than the
                # specified length.  Account for [CLS], [SEP], [SEP] with
                # "- 3"
                # option_tokens = question_tokens + option_tokens
                self._truncate_seq_pair(context_tokens_choice, option_tokens, max_seq_length - 3)
                doc_len = len(context_tokens_choice)
                if len(option_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
                    ques_len = len(option_tokens) - option_len

                tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + option_tokens + ["[SEP]"]
                segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(option_tokens) + 1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                # assert (doc_len + ques_len + option_len) <= max_seq_length
                if (doc_len + ques_len + option_len) > max_seq_length:
                    print(doc_len, ques_len, option_len, len(context_tokens_choice), len(option_tokens))
                    assert (doc_len + ques_len + option_len) <= max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))

            features.append(
                InputFeatures(
                    example_id=example.guid,
                    choices_features=choices_features,
                    label=example.label
                )
            )

        return features

    def get_dataset(self, tokenizer, max_seq_length=512, subset='train', use_cache=True):
        cache_file = './cached_dcmn_files/' + subset + '_dataset.pt'
        if use_cache and os.path.exists(cache_file):
            dataset = torch.load(cache_file)
            return dataset

        examples = self.get_examples(subset)

        features = self.convert_examples_to_features(examples, tokenizer, max_seq_length=max_seq_length)

        def select_field(features, field):
            return [
                [
                    choice[field]
                    for choice in feature.choices_features
                ]
                for feature in features
            ]

        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(features, 'option_len'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_doc_len, all_ques_len, all_option_len, all_label)

        if use_cache:
            if not os.path.exists('./cached_dcmn_files/'):
                os.mkdir('./cached_dcmn_files/')
            torch.save(dataset, cache_file)

        return dataset
