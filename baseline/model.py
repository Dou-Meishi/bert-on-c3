
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


