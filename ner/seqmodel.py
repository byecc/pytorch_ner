import torch
import torch.nn as nn
import torch.nn.functional as F
from ner.wordseq import WordSequence
from ner.crf import CRF


class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.args = data
        target_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.word_hidden = WordSequence(data)
        if data.use_crf:
            self.crf = CRF(target_size, data.use_cuda)

    def forward(self, batch_word, batch_word_len, batch_char, batch_char_len, batch_label, mask):
        out = self.word_hidden(batch_word, batch_word_len, batch_char, batch_char_len)
        if self.args.use_crf:
            loss = self.crf(out, mask, batch_label)
            score, seq = self.crf.viterbi_decode(out, mask)
        else:
            loss = F.cross_entropy(out.view(-1, out.size(2)), batch_label.view(-1), ignore_index=0, size_average=False)
            _, seq = torch.max(out, 2)
        if self.args.average_loss:
            loss = loss/self.args.batch_size
        return loss, seq
