import torch
import torch.optim as optim
import torch.autograd.variable as Var
import torch.nn.functional as F
import time
from models import *

class Trainer:
    def __init__(self,args):
        self.args = args

    def postag_lstm_train(self,train_set):
        batch_block = len(train_set) // self.args.batchsize
        if len(train_set) % self.args.batchsize:
            batch_block += 1
        model = LSTM(self.args)
        parameters = filter(lambda p:p.requires_grad,model.parameters())
        optimizer = optim.Adam(params=parameters,lr=self.args.lr)
        for i in range(self.args.epoch):
            print("No.{} training ....  ".format(i + 1), end='')
            start = time.time()
            random.shuffle(train_set)
            model.train()
            loss_sum = 0.0
            for block in range(batch_block):
                model.zero_grad()
                optimizer.zero_grad()
                left = block*self.args.batchsize
                right = left+self.args.batchsize
                if right <= len(train_set):
                    x, y, len_x = BatchGenerator.create(train_set[left:right]
                                                        ,self.args.feat_padidx,self.args.label_padidx, cuda=self.args.cuda)
                else:
                    x, y, len_x = BatchGenerator.create(train_set[left:]
                                                        ,self.args.feat_padidx, self.args.label_padidx,cuda=self.args.cuda)
                out = model.forward(x,len_x)
                loss = F.cross_entropy(out.view(-1,out.data.size()[2]),y.view(-1),ignore_index=self.args.label_padidx)
                loss.backward()
                optimizer.step()
                loss_sum += loss.data[0]
            print("loss:",loss_sum)


class BatchGenerator:

    @staticmethod
    def create(sentence_data, feat_padidx ,label_padidx,cuda=False):
        seq_max_len = 0
        num = len(sentence_data)
        for s in sentence_data:
            if seq_max_len < len(s.feature_code):
                seq_max_len = len(s.feature_code)

        sentence_data.sort(key=lambda s: len(s.feature_code), reverse=True)
        len_x = [len(s.feature_code) for s in sentence_data]
        batch_input = torch.LongTensor(num,seq_max_len)
        batch_label = torch.LongTensor(num,seq_max_len)

        for i in range(num):
            sd = sentence_data[i]
            for j in range(seq_max_len):
                if j < len(sd.feature_code):
                    batch_input[i][j] = sd.feature_code[j]
                    batch_label[i][j] = sd.label_code[j]
                else:
                    batch_input[i][j] = feat_padidx
                    batch_label[i][j] = label_padidx
        if cuda:
            batch_input = Var(batch_input).cuda()
            batch_label = Var(batch_label).cuda()
        else:
            batch_input = Var(batch_input)
            batch_label = Var(batch_label)

        return batch_input, batch_label, len_x