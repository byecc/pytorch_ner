import torch
import torch.nn as nn
import torch.nn.functional as F
from pack_embedding import LoadEmbedding


class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        self.args = data
        N = self.args.word_alphabet_size
        E = self.args.word_emb_dim
        I = self.args.hidden_dim
        if self.args.use_char:
            I += self.args.char_hidden_dim

        if self.args.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, self.args.hidden_dim, num_layers=self.args.lstm_layer,
                                batch_first=True, bidirectional=self.args.bilistm)
        elif self.args.word_feature_extractor == "GRU":
            self.gru = nn.GRU(self.input_size, self.args.hidden_dim, num_layers=self.args.lstm_layer, batch_first=True,
                              bidirectional=self.args.bilistm)
        elif self.args.word_feature_extractor == "CNN":
            pass
        else:
            print("Feature Extractor Error: don't support {} word feature extractor".format(
                self.args.word_feature_extractor))

        self.embedding = LoadEmbedding(self.args.word_alphabet_size,self.args.word_emb_dim)
        self.embedding.weight = nn.Parameter(torch.randn(N,E),requires_grad=self.args.fine_tune)
        if self.args.pretrain:
            self.embedding.load_pretrained_embedding(self.args.pretrain_file)

    def forward(self, word_inputs, word_seq_length,char_inputs,char_seq_length):
        """

             word_inputs: (batch_size,seq_len)
             word_seq_length:()
        """
        self.lstm()
