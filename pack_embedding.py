import numpy as np
from numpy import *
import torch.nn as nn
import torch
import pickle
import os
import random

random.seed(5)


class LoadEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(LoadEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.embedding_dict = {}

    def  load_pretrained_embedding(self, file, vocab_dict, embed_pickle=None, binary=False,
                                  requires_grad = False,encoding='utf8', datatype=float32):
        """
        :param lr:
        :param requires_grad: if fine tune
        :param file: pretrained embedding file path
        :param vocab_dict: features dict
        :param embed_pickle: save embed file
        :param binary: if the file is binary ,set binary True,else set False
        :param encoding: the default encoding is 'utf8'
        :param datatype: vector datatype , the default is float32
        :return:
        """
        if embed_pickle is None:
            raise FileNotFoundError
        if os.path.exists(embed_pickle):
            nparray = pickle.load(open(embed_pickle, 'rb'))
            vec_sum = np.sum(nparray[0:nparray.shape[0] - 1, :], axis=0)
            nparray[nparray.shape[0] - 1] = vec_sum / (nparray.shape[
                                                           0] - 1)  # -unknown- vector initialize by making average, -unknown- index is the last one
            print("vocabulary complete...")
            self.weight = nn.Parameter(torch.FloatTensor(nparray),requires_grad=requires_grad)
        else:
            with open(file, 'rb') as fin:
                header = str(fin.readline(), encoding).split()
                vocab_size = dim_size = 0
                if binary:
                    if header.__len__() == 2:
                        vocab_size, dim_size = int(header[0]), int(header[1])
                    else:
                        print("don't support this type")
                        exit(0)
                    binary_len = dtype(datatype).itemsize * int(dim_size)
                    for i in range(vocab_size):
                        word = []
                        while True:
                            ch = fin.read(1)
                            if ch == b' ':
                                break
                            if ch == b'':
                                raise EOFError
                            if ch != b'\n':
                                word.append(ch)
                        word = str(b''.join(word), encoding)
                        weight = fromstring(fin.read(binary_len), dtype=datatype)
                        if word in vocab_dict:
                            self.embedding_dict[word] = weight
                else:
                    if header.__len__() == 1:
                        dim_size = int(header[0])
                        vocab_size = fin.readlines().__len__() + 1
                        fin.seek(0)
                    elif header.__len__() == 2:
                        vocab_size, dim_size = int(header[0]), int(header[1])
                    else:
                        vocab_size = fin.readlines().__len__() + 1
                        dim_size = header[1:].__len__()
                        fin.seek(0)
                    for i in range(vocab_size):
                        data = str(fin.readline(), encoding).strip().split(' ')
                        word, weight = data[0], fromstring(' '.join(data[1:]), dtype=datatype, sep=' ')
                        if word in vocab_dict:
                            self.embedding_dict[word] = weight

            nparray = np.zeros((len(vocab_dict), dim_size))
            num = 0
            oov_num = 0
            oov_list = []
            for k, v in vocab_dict.items():
                if k in self.embedding_dict.keys():
                    nparray[v] = np.array(self.embedding_dict[k])
                # elif v == 0:
                #     nparray[v] = np.array([[0 for i in range(dim_size)]])
                else:
                    oov_num += 1
                    oov_list.append(k)
                    nparray[v] = np.array([[random.uniform(-0.05, 0.05) for i in range(dim_size)]])
                num += 1
                # print("word : {}".format(k))
            print("vocabulary complete...,oov num{}".format(oov_num))
            vec_sum = np.sum(nparray[0:nparray.shape[0] - 1, :], axis=0)
            nparray[nparray.shape[0] - 1] = vec_sum / (nparray.shape[
                                                           0] - 1)  # -unknown- vector initialize by making average, -unknown- index is the last one
            pickle.dump(nparray, open(embed_pickle, 'wb'))
            self.weight = nn.Parameter(torch.FloatTensor(nparray),requires_grad=requires_grad)
        return torch.FloatTensor(nparray)

