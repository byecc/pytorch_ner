import torch
import torch.nn as nn
import torch.nn.functional as F
from pack_embedding import LoadEmbedding
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
import numpy as np


class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        self.args = data

        # word
        alp_length = data.word_alphabet_size
        emb_dim = data.word_embed_dim
        hidden_dim = data.hidden_dim

        if data.word_feature_extractor == "LSTM":
            self.word_feature = nn.LSTM(emb_dim, data.hidden_dim, num_layers=data.lstm_layer,
                                        batch_first=True, bidirectional=data.bilstm)
        elif data.word_feature_extractor == "GRU":
            self.word_feature = nn.GRU(emb_dim, data.hidden_dim, num_layers=data.lstm_layer, batch_first=True,
                                       bidirectional=self.args.bilistm)
        elif data.word_feature_extractor == "CNN":
            pass
        else:
            print("Feature Extractor Error: don't support {} word feature extractor".format(
                self.args.word_feature_extractor))

        self.word_embedding = LoadEmbedding(data.word_alphabet_size, data.word_embed_dim)
        if data.pretrain:
            self.word_embedding.load_pretrained_embedding(data.word_embed_path, data.label_alphabet.instance2index,
                                                          requires_grad=data.fine_tune,
                                                          embed_pickle=data.word_embed_save, binary=False)
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet_size, data.word_embed_dim)))

        # char
        if data.use_char:
            if data.bilstm:
                hidden_dim += self.args.char_hidden_dim
            hidden_dim += self.args.char_hidden_dim
            self.char_embedding = nn.Embedding(data.char_alphabet_size, data.char_embed_dim)
            if data.pretrain_char_embedding is not None:
                self.char_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_char_embedding))
            else:
                self.char_embedding.weight.data.copy_(
                    torch.from_numpy(self.random_embedding(data.char_alphabet_size, data.char_embed_dim)))
            if data.char_feature_extractor == "LSTM":
                self.char_feature = nn.LSTM(data.char_embed_dim, data.char_hidden_dim, num_layers=1,
                                            batch_first=True, bidirectional=data.bilstm)
            if data.char_feature_extractor == "GRU":
                self.char_feature = nn.GRU(data.char_embed_dim, data.char_hidden_dim, num_layers=1,
                                           batch_first=True, bidirectional=data.bilstm)
            self.char2m = nn.Linear(data.char_hidden_dim * 2, data.char_hidden_dim * 2, bias=False)

        self.drop = nn.Dropout(data.dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, data.label_alphabet_size)

        # attention
        if data.attention:
            self.attn1 = nn.Linear(data.word_embed_dim, data.attention_dim)
            if data.bilstm:
                self.attn2 = nn.Linear(data.char_hidden_dim * 2, data.attention_dim, bias=False)
            else:
                self.attn2 = nn.Linear(data.char_hidden_dim, data.attention_dim, bias=False)
            self.attn3 = nn.Linear(data.attention_dim, 1, bias=False)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_inputs, word_seq_length, char_inputs, char_seq_length):
        """

             word_inputs: (batch_size,seq_len)
             word_seq_length:()
        """
        word_emb = self.word_embedding(word_inputs)
        word_emb = pack(word_emb, word_seq_length.numpy(), batch_first=True)
        word_lstm_out, word_hidden = self.word_feature(word_emb)
        word_lstm_out = pad(word_lstm_out, batch_first=True)
        word_rep = word_lstm_out[0]
        if self.args.use_char:
            size = char_inputs.size(0)
            char_emb = self.char_embedding(char_inputs)
            char_emb = pack(char_emb, char_seq_length.numpy(), batch_first=True)
            char_lstm_out, char_hidden = self.char_feature(char_emb)
            char_lstm_out = pad(char_lstm_out, batch_first=True)
            char_hidden = char_hidden[0].transpose(1, 0).contiguous().view(size, -1)
            char_hidden = F.tanh(self.char2m(char_hidden))
            char_hidden = char_hidden.view(word_emb.size()[0], word_emb.size()[1], -1)
            if self.args.attention:
                word_rep = F.tanh(self.attn1(word_emb) + self.attn2(char_hidden))
                z = F.sigmoid(self.attn3(word_rep))
                word_rep = F.mul(z * word_emb) + F.mul((1 - z) * char_hidden)
            else:
                word_rep = torch.cat((word_emb, char_hidden), 2)

        out = self.hidden2tag(self.drop(word_rep))
        return out
