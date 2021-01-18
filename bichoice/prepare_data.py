'''
create csv data for training.
'''

import json
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

# modules in this directory
# NOTE: You can only run this script in its directory.
#       Call from outside causes following import commands fails
import utils
from data_processor import (
    C3BinaryExample,
    C3BinaryDataProcessor,
)


def C3Example_to_C3BinaryExample(eList):
    '''
    create `C3BinaryExample`s from `C3Example`s.
    
    Args
    ----
    `eList` : a list of `bichoice.utils.C3Example`
    
    Return
    ------
    `out` : a list of `bichioce.data_processor.C3BinaryExample`.
    '''
    out = []
    for e in eList:
        passage = ''.join(e.sentences)
        question = e.question
        answer = e.options[e.label]
        for o in e.options:
            if o == answer:
                continue
            out.append(C3BinaryExample(passage, question, answer, o, 0))
            out.append(C3BinaryExample(passage, question, o, answer, 1))
    return out


def C3BinaryExamples_to_dataframe(eList):
    '''
    convert a list of `C3BinaryExample` to `pandas.DataFrame`
    
    Args
    ----
    `eList` : a list of `C3BinaryExample`
    
    Return
    ------
    `out` : `pandas.DataFrame` 
    '''
    out = {
        'passage': [e.passage for e in eList],
        'question': [e.question for e in eList],
        'choice_0': [e.choice_0 for e in eList],
        'choice_1': [e.choice_1 for e in eList],
        'label': [e.label for e in eList],
    }
    return pd.DataFrame(out)


def show_c3_binary_example(e):
    '''show all info of a single `C3BinaryExample`'''
    print('-----PASSAGE-----')
    print(e.passage)
    print('-----QUESTION-----')
    print(e.question)
    print('-----CHOICE_0-----')
    print(e.choice_0)
    print('-----CHOICE_1-----')
    print(e.choice_1)
    print('-----LABEL-----')
    print(e.label)


def main():
    # declare a namespace
    D = utils.GlobalSettings({
            'DATADIR': '../data/',
            'OUTDIR': '../outputs/csv-data/',
        })

    if not os.path.exists(D.OUTDIR):
        os.makedirs(D.OUTDIR)

    train = utils.get_all_C3examples(os.path.join(D.DATADIR, 'c3-m-train.json'))
    train+= utils.get_all_C3examples(os.path.join(D.DATADIR, 'c3-d-train.json'))

    valid = utils.get_all_C3examples(os.path.join(D.DATADIR, 'c3-m-dev.json'))
    valid+= utils.get_all_C3examples(os.path.join(D.DATADIR, 'c3-d-dev.json'))

    test = utils.get_all_C3examples(os.path.join(D.DATADIR, 'c3-m-test.json'))
    test+= utils.get_all_C3examples(os.path.join(D.DATADIR, 'c3-d-test.json'))

    bi_train_df = C3BinaryExamples_to_dataframe(C3Example_to_C3BinaryExample(train))
    bi_valid_df = C3BinaryExamples_to_dataframe(C3Example_to_C3BinaryExample(valid))
    bi_test_df = C3BinaryExamples_to_dataframe(C3Example_to_C3BinaryExample(test))

    print('check an example')

    show_c3_binary_example(bi_train_df.loc[random.choice(bi_train_df.index)])

    bi_train_df.to_csv(os.path.join(D.OUTDIR, 'binary-train.csv'),
                        index=False, encoding='utf-8')
    bi_valid_df.to_csv(os.path.join(D.OUTDIR, 'binary-dev.csv'),
                        index=False, encoding='utf-8')
    bi_test_df.to_csv(os.path.join(D.OUTDIR, 'binary-test.csv'),
                        index=False, encoding='utf-8')

    print('created csv files saved to', D.OUTDIR)


if __name__ == '__main__':
    main()
