import torch
import argparse
import pack_embedding
from ner.datadef import *
import torch.optim as optim
import random
import numpy as np
from ner.seqmodel import SeqModel

seed_num = 233
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def train(data):
    print("Training model...")
    model = SeqModel(data)
    if data.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum, weight_decay=data.weight_decay)
    elif data.optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.lr, weight_decay=data.weight_decay)
    elif data.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=data.lr, weight_decay=data.weight_decay)
    else:
        print("Optimizer Error: {} optimizer is not support.".format(data.optimizer))
    for idx in range(data.iter):
        train_num = len(data.train_text)
        batch_block = train_num // data.batch_size + 1
        for block in range(batch_block):
            left = block * data.batch_size
            right = left + data.batch_size
            if right > train_num:
                right = train_num
            instance = data.train_idx[left:right]
            generate_batch(instance,data.use_cuda)


def generate_batch(instance, gpu):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiLSTM+CRF")
    parser.add_argument('--config', help="Config File")

    args = parser.parse_args()
    data = Data()
    data.read_config(args.config)
    status = data.status
    # data.show_config()

    if status.lower() == "train":
        data.get_instance("train")
        data.get_instance("dev")
        data.get_instance("test")
        data.build_alphabet()
        data.get_instance_index("train")
        data.get_instance_index("dev")
        data.get_instance_index("test")
        train(data)
