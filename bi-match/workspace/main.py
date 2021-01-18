import math
import os
import json

import random
from tqdm import tqdm, trange

import numpy as np
import torch
from apex import amp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers.models.bert import BertTokenizer
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dcmn_data_processor import DCMNDataProcessor
from data_processor import DataProcessor
from dcmn_model import BertForMultipleChoiceWithMatch
from model import BertForClassification, MyModel


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(config, model, train_dataset, eval_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], sampler=train_sampler)

    t_total = len(train_dataloader) // config["gradient_accumulation_steps"] * config["num_train_epochs"]
    num_warmup_steps = math.floor(t_total * 0.1)

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
    ]

    # optimizer = AdamW(optimizer_parameters, lr=config["learning_rate"], eps=1e-8)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    if config["fp16"] == 1:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    nb_tr_examples, nb_tr_steps = 0, 0
    model.zero_grad()
    train_iterator = trange(int(config["num_train_epochs"]), desc="Epoch", disable=True)
    set_seed(config["seed"])
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", leave=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.cuda() for t in batch)
            # inputs = {'input_ids':      batch[0],
            #           'attention_mask': batch[1],
            #           'token_type_ids': batch[2],
            #           'labels':         batch[3]}
            outputs = model(*batch)
            loss = outputs[0]
            
            if config["gradient_accumulation_steps"] > 1:
                loss = loss / config["gradient_accumulation_steps"]

            if config["fp16"] == 1:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config["max_grad_norm"])
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

            tr_loss += loss.item()
            logging_loss += loss.item()
            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                stat = 'epoch {} | step {} | lr {:.6f} | loss {:.6f}'.format(epoch, global_step, scheduler.get_last_lr()[0], logging_loss)
                epoch_iterator.set_postfix_str(str(stat))
                logging_loss = 0.0

        # Save model checkpoint
        eval_loss, eval_metric = evaluate(config, model, eval_dataset)
        print("epoch: {}, eval_result: {:.6f}, eval_loss: {:.6f}".format(epoch, eval_metric, eval_loss))
        save_dir = os.path.join(config["save_dir"], 'checkpoint-{}'.format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'model.bin'))
        torch.save(config, os.path.join(save_dir, 'training_args.bin'))
        print("Saving model checkpoint to {}".format(save_dir))

    return global_step, tr_loss / global_step


def evaluate(config, model, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["eval_batch_size"], sampler=eval_sampler)

    eval_loss, eval_accuracy = 0.0, 0.0
    nb_eval_steps, nb_eval_examples = 0, 0

    for _, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", leave=False)):
        model.eval()
        batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            # inputs = {'input_ids': batch[0],
            #           'attention_mask': batch[1],
            #           'token_type_ids': batch[2],
            #           'labels': batch[3] if len(batch) == 4 else None}
            outputs = model(*batch)
            tmp_eval_loss, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()
        label_ids = batch[-1].cpu().numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += batch[0].size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_loss, eval_accuracy


def main():
    config = json.load(open('config.json', 'r'))

    set_seed(config["seed"])

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    # model_config = transformers.BertConfig.from_pretrained(config["model_name"])
    # tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    # model = MyModel(config["model_name"])
    model = BertForMultipleChoiceWithMatch(config["model_name"])
    # model = AutoModelForMultipleChoice.from_pretrained(config["model_name"])

    model.cuda()

    # processor = DataProcessor(config["data_dir"])
    processor = DCMNDataProcessor(config["data_dir"])

    # train_examples = processor.get_train_examples()
    # train_dataset = processor.get_dataset(train_examples, tokenizer, config["max_length"])
    #
    # valid_examples = processor.get_dev_examples()
    # valid_dataset = processor.get_dataset(valid_examples, tokenizer, config["max_length"])
    #
    # test_examples = processor.get_test_examples()
    # test_dataset = processor.get_dataset(test_examples, tokenizer, config["max_length"])

    train_dataset = processor.get_dataset(tokenizer, config["max_length"], subset='train')
    valid_dataset = processor.get_dataset(tokenizer, config["max_length"], subset='valid')
    test_dataset = processor.get_dataset(tokenizer, config["max_length"], subset='test')

    train(config, model, train_dataset, valid_dataset)
    result = evaluate(config, model, test_dataset)
    print(result[:2])


if __name__ == '__main__':
    main()
