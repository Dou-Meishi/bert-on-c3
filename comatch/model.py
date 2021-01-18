
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModel
from transformers.models.bert import BertModel


class BertForClassification(nn.Module):
    """
        这是一个多分类问题，也就是4个中选概率最大的，相比于二分类，应该更适合这个任务的
    """
    def __init__(self, model_name_or_path, dropout=0.1):
        super().__init__()
        # self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.bert = BertModel.from_pretrained(model_name_or_path)

        # self.bert.requires_grad_(True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.n_class = 4

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):

        seq_length = input_ids.size(2)

        bert_output = self.bert(
            input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length),
            token_type_ids=token_type_ids.view(-1, seq_length),
        )

        pooler_output = bert_output[1]
        pooler_output = self.dropout(pooler_output)
        logits = self.linear(pooler_output).view(-1, self.n_class)
        # logits = F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            # loss = F.nll_loss(logits, labels, reduction='sum')
            return loss, logits

        return None, logits

class MyModel(nn.Module):
    def __init__(self, model_name_or_path, dropout=0.1):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 1)
        self.dropout = dropout

    def forward(self, input_ids_a, attention_mask_a, token_type_ids_a, input_ids_b, attention_mask_b, token_type_ids_b, labels=None):
        batch_size = input_ids_a.size(0)
        # batch_size * seqlen * 768
        doc_output = self.bert(input_ids_a, attention_mask=attention_mask_a, token_type_ids=token_type_ids_a)[0]
        # doc_output = self.bert(input_ids_a, attention_mask=attention_mask_a, token_type_ids=token_type_ids_a)[1] # batch_size * 768

        ques_seq_len = input_ids_b.size(-1)
        choices_outputs = self.bert(
            input_ids_b.view(-1, ques_seq_len),
            attention_mask=attention_mask_b.view(-1, ques_seq_len),
            token_type_ids=token_type_ids_b.view(-1, ques_seq_len),
        )[1] # (batch * 4, 768)
        choices_outputs = choices_outputs.view(-1, 4, 768) # (batch size * 4 * 768)

        score = torch.matmul(choices_outputs, doc_output.transpose(-1, -2)) # batch_size * 4 * seqlen
        score = F.softmax(score, -1) # batch_size * 4 * seqlen

        output = torch.matmul(score, doc_output) # batch_size * 4 * 768
        choices_outputs = self.linear1(choices_outputs)

        output = F.relu(output * choices_outputs)
        output = self.linear2(output)

        # output = output / torch.norm(output, dim=-1, keepdim=True).sqrt()
        # choices_outputs = self.linear2(choices_outputs)
        # choices_outputs = choices_outputs / torch.norm(choices_outputs, dim=-1, keepdim=True).sqrt()
        # sims = (output * choices_outputs).sum(dim=-1)
        # output = F.dropout(output, self.dropout)

        # output = torch.matmul(doc_output.view(-1, 1, 768), choices_outputs.transpose(-1, -2))
        # logits = output.view(-1, 4)

        logits = output.view(-1, 4)
        # logits = self.linear(output).view(-1, 4)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            return loss, logits

        return None, logits