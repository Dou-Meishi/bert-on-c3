
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.models.bert import BertModel, BertForPreTraining

def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask.requires_grad_ = False
    # mask = None
    if mask is None:
        result = F.softmax(vector, dim=-1)
    else:
        result = F.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result

class FuseNet(nn.Module):
    def __init__(self, hidden_size):
        super(FuseNet, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear1(q)
        lp = self.linear2(p)
        mid = F.sigmoid(lq+lp)
        output = p * mid + q * (1-mid)
        return output

class SSingleMatchNet(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(SSingleMatchNet, self).__init__()
        self.trans_linear1 = nn.Linear(hidden_size, hidden_size)
        self.trans_linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear1(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = masked_softmax(att_weights, seq_len)
        att_norm = F.dropout(att_norm, p=self.dropout)

        att_vec = att_norm.bmm(proj_q)
        output = F.relu(self.trans_linear2(att_vec))
        return output

def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i]+1]
        doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
        ques_option_seq_output[i, :ques_len[i]+option_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[i] + 1]
        option_seq_output[i, :option_len[i]] = sequence_output[i,
                                                 doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
                                                   i] + 2]
    return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output


class BertForMultipleChoiceWithMatch(nn.Module):

    def __init__(self, model_name_or_path, hidden_size=768, dropout=0.1, num_choices=4):
        super(BertForMultipleChoiceWithMatch, self).__init__()
        self.num_choices = num_choices
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(hidden_size, 1)
        # self.classifier2 = nn.Linear(2 * hidden_size, 1)
        self.classifier3 = nn.Linear(3 * hidden_size, 1)
        # self.classifier4 = nn.Linear(4 * hidden_size, 1)
        # self.classifier6 = nn.Linear(6 * hidden_size, 1)
        self.ssmatch = SSingleMatchNet(hidden_size, dropout)
        self.fuse = FuseNet(hidden_size)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None,
                option_len=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()

        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        sequence_output, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)[:2]

        doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output = seperate_seq(
            sequence_output, doc_len, ques_len, option_len)

        # pooled_ques_option = ques_option_seq_output.mean(1, keepdim=False).view(-1, 1, 768)

        # score = torch.matmul(pooled_ques_option, doc_seq_output.transpose(-1, -2))  # batch_size * 4 * seqlen
        # score = F.softmax(score, -1)  # batch_size * 4 * seqlen

        # doc_seq_output = torch.matmul(score, doc_seq_output)

        pa_output = self.ssmatch([doc_seq_output, option_seq_output, option_len + 1])
        ap_output = self.ssmatch([option_seq_output, doc_seq_output, doc_len + 1])
        pq_output = self.ssmatch([doc_seq_output, ques_seq_output, ques_len + 1])
        qp_output = self.ssmatch([ques_seq_output, doc_seq_output, doc_len + 1])
        qa_output = self.ssmatch([ques_seq_output, option_seq_output, option_len + 1])
        aq_output = self.ssmatch([option_seq_output, ques_seq_output, ques_len + 1])

        pa_output_pool, _ = pa_output.max(1)
        ap_output_pool, _ = ap_output.max(1)
        pq_output_pool, _ = pq_output.max(1)
        qp_output_pool, _ = qp_output.max(1)
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)

        pa_fuse = self.fuse([pa_output_pool, ap_output_pool])
        pq_fuse = self.fuse([pq_output_pool, qp_output_pool])
        qa_fuse = self.fuse([qa_output_pool, aq_output_pool])

        cat_pool = torch.cat([pa_fuse, pq_fuse, qa_fuse], 1)
        output_pool = self.dropout(cat_pool)
        match_logits = self.classifier3(output_pool)
        match_reshaped_logits = match_logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return None, match_reshaped_logits