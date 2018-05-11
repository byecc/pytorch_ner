import torch
import torch.nn as nn
from torch.autograd import Variable as Var
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from pack_embedding import *

class LSTM(nn.Module):
    def __init__(self,args):
        super(LSTM,self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_dim
        N = args.num_layers
        B = args.batchsize
        L = args.label_num

        self.embedding = LoadEmbedding(V,D)
        if args.pretrain:
            pass
        else:
            self.embedding.weight = nn.Parameter(torch.randn((V, D)), requires_grad=args.tuned)
        self.lstm = nn.LSTM(D,H,N,batch_first=True)
        self.hidden = self.init_hidden(N,B,H,cuda = args.cuda)
        self.hidden2label = nn.Linear(H,L)
        self.dropout = nn.Dropout(args.dropout)

    def init_hidden(self,num_layers,batch_size,hidden_dim,cuda=False):
        if cuda:
            return (Var(torch.zeros(num_layers, batch_size, hidden_dim)).cuda(),
                    Var(torch.zeros(num_layers, batch_size, hidden_dim)).cuda())
        else:
            return (Var(torch.zeros(num_layers, batch_size, hidden_dim)),
                    Var(torch.zeros(num_layers, batch_size, hidden_dim)))


    def forward(self, sentence,sen_len):
        embed = self.embedding(sentence)
        embed = pack(embed,sen_len,batch_first=True)
        lstm_out,self.hidden = self.lstm(embed)
        lstm_out,_ =unpack(lstm_out,batch_first=True)
        lstm_out = self.dropout(lstm_out)
        out = self.hidden2label(lstm_out)
        return out


