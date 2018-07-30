import torch
import argparse
import pack_embedding
from ner.datadef import *
import torch.optim as optim
import torch.autograd as autograd
import torch.functional as F
import random
import numpy as np
import time
from ner.seqmodel import SeqModel
from ner.eval import *

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
        epoch_start = temp_start = time.time()
        train_num = len(data.train_text)
        batch_block = train_num // data.batch_size + 1
        correct = total = total_loss = 0
        # random.shuffle(data.train_idx)
        for block in range(batch_block):
            left = block * data.batch_size
            right = left + data.batch_size
            if right > train_num:
                right = train_num
            instance = data.train_idx[left:right]
            batch_word, batch_word_len, batch_char, batch_char_len, batch_label, mask = generate_batch(instance,
                                                                                                       data.use_cuda)
            loss, seq = model.forward(batch_word, batch_word_len, batch_char, batch_char_len, batch_label, mask)
            right_token, total_token = predict_check(seq, batch_label, mask)
            correct += right_token
            total += total_token
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            model.zero_grad()
        epoch_end = time.time()
        print("Finish  Epoch:{}. Time:{}. Loss:{}. acc:{}".format(idx, epoch_end - epoch_start, total_loss,
                                                                  correct / total))
        evaluate(data, model, "dev")
        evaluate(data, model, "test")


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_idx
    elif name == "dev":
        instances = data.dev_idx
    elif name == "test":
        instances = data.test_idx
    else:
        print("Error: no {} evaluate data".format(name))
    batch_size = data.batch_size
    num = len(instances)
    batch_block = num // batch_size + 1
    correct_num = gold_num = pred_num = 0
    for batch in range(batch_block):
        left = batch_block * batch_size
        right = left + batch_size
        if right > num:
            right = num
        train_instance = instances[left:right]
        batch_word, batch_word_len, batch_char, batch_char_len, batch_label, mask = generate_batch(instances,
                                                                                                   data.use_cuda)
        loss, seq = model(batch_word, batch_word_len, batch_char, batch_char_len, batch_label, mask)
        gold_list, pred_list = seq_eval(data, seq, batch_label)
        pred, correct, gold = get_ner_measure(pred_list, gold_list, data.tag_scheme)
        correct_num += correct
        pred_num += pred
        gold_num += gold
    precision = pred_num / correct_num
    recall = pred_num / gold_num
    f = 2 * precision * recall / (precision + recall)
    print("{} Eval: \tp:{}\tr:{}\tf:{}".format(name, precision, recall, f))


def predict_check(predict, gold, mask):
    predict = predict.cpu().data.numpy()
    gold = gold.cpu().data.numpy()
    mask = mask.cpu().data.numpy()
    result = (predict == gold)
    right_token = np.sum(result * mask)
    total_token = mask.sum()
    return right_token, total_token


def generate_batch(instance, gpu):
    """

    :param instance:  [[word,char,label],...]
    :return:
        zero padding for word and char ,with  their batch length

    """
    batch_size = len(instance)
    words = [ins[0] for ins in instance]
    chars = [ins[1] for ins in instance]
    labels = [ins[2] for ins in instance]
    word_seq_length = torch.LongTensor(list(map(len, words)))
    max_seq_length = word_seq_length.max()
    word_seq_tensor = autograd.Variable(torch.zeros(batch_size, max_seq_length), requires_grad=False).long()
    label_seq_tensor = autograd.Variable(torch.zeros(batch_size, max_seq_length), requires_grad=False).long()
    mask = autograd.Variable(torch.zeros(batch_size, max_seq_length)).byte()
    for idx, (seq, label, seq_len) in enumerate(zip(words, labels, word_seq_length)):
        word_seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seq_len] = torch.LongTensor(label)
        mask[idx, :seq_len] = torch.Tensor([1] * seq_len)
    word_seq_length, sort_idx = torch.sort(word_seq_length, descending=True)
    word_seq_tensor = word_seq_tensor[sort_idx]
    label_seq_tensor = label_seq_tensor[sort_idx]
    mask = mask[sort_idx]

    # char
    pad_chars = [chars[idx] + [[0]] * (max_seq_length - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros(batch_size, max_seq_length, max_word_len)).long()
    char_seq_length = torch.LongTensor(length_list)
    for idx, (seq, seq_len) in enumerate(zip(pad_chars, char_seq_length)):
        for idy, (word, word_len) in enumerate(zip(seq, seq_len)):
            char_seq_tensor[idx, idy, :word_len] = torch.LongTensor(word)
    # char_seq_tensor = char_seq_tensor[sort_idx]
    # char_seq_length = char_seq_length[sort_idx]
    char_seq_tensor = char_seq_tensor.view(-1, max_word_len)
    char_seq_length = char_seq_length.view(-1)
    char_seq_length, char_sort_idx = torch.sort(char_seq_length, descending=True)
    char_seq_tensor = char_seq_tensor[char_sort_idx]

    return word_seq_tensor, word_seq_length, char_seq_tensor, char_seq_length, label_seq_tensor, mask


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
        data.build_pretrain_emb()
        train(data)
