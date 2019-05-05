import torch
import torch.nn as nn
import torch.nn.functional as F
from wordseq import WordSequence
from crf import CRF
from hyperlstm import *
from multi_hyperlstm import *


class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.args = data
        target_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        if data.hyperlstm:
            #self.word_hidden = HyperLSTM(data)
            self.word_hidden = MultiHyperLSTM(data)
        else:
            self.word_hidden = WordSequence(data)
        if data.use_crf:
            self.crf = CRF(target_size, data.use_cuda)

    def neg_log_likehood(self, batch_word, batch_feat,batch_word_len, batch_char, batch_char_len, char_recover, batch_label,mask,batch_dict,batch_bert):
        out = self.word_hidden(batch_word,batch_feat, batch_word_len, batch_char, batch_char_len,char_recover,batch_dict,mask,batch_bert)
        if self.args.use_crf:
            loss = self.crf(out, mask, batch_label)
            score, seq = self.crf.viterbi_decode(out, mask)
        else:
            batch_size = out.size(0)
            seq_len = out.size(1)
            loss_function = nn.NLLLoss(ignore_index=0, reduction="sum")
            out = out.view(-1,out.size(2))
            score = F.log_softmax(out,1)
            loss = loss_function(score,batch_label.view(-1))
            _, seq = torch.max(score, 1)
            seq = seq.view(batch_size,seq_len)
            seq = mask.long() * seq
        if self.args.average_loss:
            loss = loss / self.args.batch_size
        return loss, seq

    def forward(self,batch_word, batch_feat,batch_word_len, batch_char, batch_char_len, char_recover, batch_label, mask,batch_dict,batch_bert):
        out = self.word_hidden(batch_word, batch_feat,batch_word_len, batch_char, batch_char_len, char_recover,batch_dict,mask,batch_bert)
        if self.args.use_crf:
            score, seq = self.crf.viterbi_decode(out, mask)
        else:
            batch_size = out.size(0)
            seq_len = out.size(1)
            out = out.view(-1,out.size(2))
            _,seq = torch.max(out,1)
            seq = seq.view(batch_size,seq_len)
            seq = mask.long()*seq
        return seq
