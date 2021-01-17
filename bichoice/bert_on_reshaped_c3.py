'''
Apply a simple binary sequence classifier on reshaped C3 dataset
'''
import json
import os

import torch
import transformers
from tqdm import tqdm

# modules in current directory
from . import utils
from .data_processor import (
    C3BinaryExample,
    C3BinaryDataProcessor
)


def compare(D, model, processor, passage, question, c1, c2, avg=True):
    '''
    compare two options and select one with the higher probability.

    Args
    ----
    `D` : instance of `GlobalSettings`
    
    `model` : binary classifier

    `processor` : instance of `C3BinaryDataProcessor`

    `passage` : str

    `question` : str

    `c1` : str

    `c2` : str

    `avg` : bool, if `True`, the we switch the position of `c1` and `c2` and average output as probability. 

    Return
    ------
    `label` : int, 0 means selecting c1, 1 means selecting c2
    '''
    # tokenizing
    example1 = C3BinaryExample(passage, question, c1, c2)
    example2 = C3BinaryExample(passage, question, c2, c1)
    f1 = processor.convert_example_to_features(example1)
    f2 = processor.convert_example_to_features(example2)
    batch = {
'input_ids': torch.LongTensor([f1.input_ids, f2.input_ids]),
'attention_mask': torch.LongTensor([f1.input_mask,f2.input_mask]),
'token_type_ids': torch.LongTensor([f1.segment_ids,f2.segment_ids]),
    }
    for k in batch:
        batch[k] = batch[k].to(D.DEVICE)

    model.to(D.DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(**batch)
        logits = output.logits
    if avg:
        logit1 = (logits[0,0] + logits[1,1]).item()
        logit2 = (logits[0,1] + logits[1,0]).item()
    else:
        logit1 = logits[0,0].item()
        logit2 = logits[0,1].item()

    return int(logit2 > logit1)


def evaluate_original_set(D, model, processor, fn):
    '''
    get accuracy on original dataset (before reshaping)

    Args
    ----
    `D` : instance of `GlobalSettings`

    `model` : binary classifier

    `processor` : instance of `C3BinaryDataProcessor`

    `fn` : path to original json file

    Return
    ------
    `acc` : accuracy in decimal
    '''
    eList = utils.get_all_C3examples(fn)
    n, N = 0, len(eList)
    for e in tqdm(eList, desc='evaluating'):
        candidate = e.options[0]
        for c2 in e.options[1:]:
            bin_label = compare(D, model, processor, ''.join(e.sentences), e.question, candidate, c2)
            if bin_label == 1:
                candidate = c2 # out model prefer c2 than candidate
        if candidate == e.options[e.label]:
            n += 1
    return n/N



def main():
    # --------------------PREPARE--------------------
    D = utils.GlobalSettings.from_json('bert_on_reshaped_c3.json')

    if D.FP16: # use mix precision
        from apex import amp

    if not os.path.exists(D.OUTDIR):
        os.mkdir(D.OUTDIR)

    assert all([
        os.path.exists(D.TRAIN_SET),
        os.path.exists(D.VALID_SET),
        os.path.exists(D.VALID_SET_JSON),
        os.path.exists(D.TEST_SET_JSON),
    ])

    D.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = transformers.BertForSequenceClassification.from_pretrained(
        D.MODEL_NAME, num_labels=2)
    model.to(D.DEVICE)

    # --------------------GET DATASET--------------------
    # do NOT use AutoTokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(D.MODEL_NAME)
    processor = C3BinaryDataProcessor(tokenizer, D.MAX_LENGTH)

    train_dataset = processor.get_dataset(D.TRAIN_SET, with_label=True)
    valid_dataset = processor.get_dataset(D.VALID_SET, with_label=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=D.BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=D.BATCH_SIZE, shuffle=True)
    num_train_steps = D.EPOCHS*(len(train_dataloader)//D.ACCUMULATION_STEPS+1)

    # --------------------TRAINING--------------------
    optimizer = transformers.AdamW(model.parameters(), lr=D.LR)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, D.WARMUP_STEPS, num_train_steps)

    if D.FP16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    model.zero_grad()
    training_stats = []

    for epoch in range(D.EPOCHS):

        print(f'\nTraining epoch {epoch+1}/{D.EPOCHS}')
        model.train()
        tr_loss = 0.0
        for i, b in enumerate(tqdm(train_dataloader)):
            b = tuple(t.to(D.DEVICE) for t in b)
            output = model(input_ids=b[0], attention_mask=b[1], token_type_ids=b[2], labels=b[3])
            loss = output.loss/D.ACCUMULATION_STEPS
            tr_loss += output.loss.item()

            if D.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_parameters(optimizer), D.MAX_GRAD_NORM)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), D.MAX_GRAD_NORM)

            if (i+1)%D.ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        tr_loss /= len(train_dataloader)
        print('Average training loss:', tr_loss)

        print(f'\nValidation epoch {epoch+1}/{D.EPOCHS}')
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for i, b in enumerate(tqdm(valid_dataloader)):
                b = tuple(t.to(D.DEVICE) for t in b)
                output = model(input_ids=b[0], attention_mask=b[1], token_type_ids=b[2], labels=b[3])
                va_loss += output.loss.item()

        va_loss /= len(valid_dataloader)
        print('Average validation loss:', va_loss)

        print('\ncalculating accuracy')
        valid_acc = evaluate_original_set(D, model, processor, D.VALID_SET_JSON)
        test_acc = evaluate_original_set(D, model, processor, D.TEST_SET_JSON)
        training_stats.append({
            'epoch': epoch+1,
            'training loss': tr_loss,
            'validation loss': va_loss,
            'accuracy on validation set': valid_acc,
            'accuracy on test set': test_acc,
        })

        print('\n',training_stats[-1])

        # save output
        save_dir = os.path.join(D.OUTDIR, f'epoch-{epoch+1}')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'model.bin'))
        D.save(os.path.join(save_dir, 'global_settings.json'))
        with open(os.path.join(save_dir, 'training_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, indent=4, ensure_ascii=False)
        print('output saved to', save_dir)

    print('DONE')
    return D, model, processor


if __name__ == '__main__':
    main()
