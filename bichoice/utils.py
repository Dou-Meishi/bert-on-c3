import json
import random


class GlobalSettings(object):
    '''a namespace for hyper parameters'''
    def __init__(self, d):
        self.__dict__.update(d)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, ensure_ascii=False)

    def update(self, d):
        self.__dict__.update(d)


    def save(self, fn, **kws):
        '''
        save parameter settings to json file `fn`.

        Args
        ----
        `fn` : path like object
        '''
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False, **kws)


    @classmethod
    def from_json(cls, fn):
        '''
        Args
        ----
        `fn` : path to a json file.
        '''
        with open(fn, 'r', encoding='utf-8') as f:
            return cls(json.load(f))



class C3Example(object):
    '''
    a multiple choice example in C3 dataset.
    '''
    def __init__(self, sentences, question, options, label, _id):
        self.sentences = sentences # list of str
        self.question = question # str
        self.options = options # list of str
        self.label = label # index of the correct option
        self._id = _id

        if label is not None:
            assert label in list(range(len(options)))


    def __str__(self):
        return str(dict({
            'sentences': self.sentences,
            'question': self.question,
            'options': self.options,
            'label': self.label if self.label is not None else 'NA'
        }))



def get_all_C3examples(dataset_fn):
    '''
    Load all examples from json file `dataset_fn`.
    Return is a `list` of `C3Example`.
    '''
    with open(dataset_fn, 'r', encoding='utf-8') as fpr:
        dataList = json.load(fpr)
    exampleList = []
    for data in dataList:
        # parse data point to several `C3Example`s
        sentences = data[0]
        idprefix = data[2]
        for i, ith in enumerate(data[1]):
            question = ith['question']
            options = ith['choice']
            label = options.index(ith['answer'])
            exampleList.append(C3Example(
                sentences, question, options, label, idprefix+f':{i}'
            ))

    return exampleList



class RandomBot(object):
    '''a random bot for C3'''

    def __call__(self, example):
        return random.choice(list(range(len(example.options))))



class CountBot(object):
    '''a naive bot for C3 by counting'''

    def __call__(self, example):
        '''predict label of example'''
        scores = []
        for option in example.options:
            d = ''.join(example.sentences) + example.question
            scores.append(d.count(option))
        best_idx = [i for i, s in enumerate(scores) if s==max(scores)]
        return random.choice(best_idx)



def test_bot(bot, fn):
    correct_examples = []
    incorrect_examples = []
    for e in get_all_C3examples(fn):
        pred = bot(e)
        if pred == e.label:
            correct_examples.append(e)
        else:
            incorrect_examples.append(e)
    return correct_examples, incorrect_examples
