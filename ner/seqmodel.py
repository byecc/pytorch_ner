import torch
import torch.nn as nn
import torch.nn.functional as F
from ner.wordseq import WordSequence
from ner.crf import CRF


class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.args = data
        self.word_hidden = WordSequence(data)
        if self.args.use_crf:
            self.crf = CRF(self.args.label_alphabet_size, self.args.use_cuda)

    def forward(self, *input):
        pass
