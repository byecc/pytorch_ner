import torch
import argparse
import pack_embedding
from data import *
import torch.optim as optim
import torch.autograd as autograd
import torch.functional as F
import random
import numpy as np
import time
from seqmodel import SeqModel
from eval import *

seed_num = 233
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def train(data):
    print("Training model...")
    model = SeqModel(data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if data.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=data.lr, momentum=data.momentum, weight_decay=data.weight_decay)
    elif data.optimizer == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=data.lr, weight_decay=data.weight_decay)
    elif data.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=data.lr, weight_decay=data.weight_decay)
    else:
        print("Optimizer Error: {} optimizer is not support.".format(data.optimizer))
    for idx in range(data.iter):
        epoch_start = temp_start = time.time()
        train_num = len(data.train_text)
        if train_num % data.batch_size == 0:
            batch_block = train_num // data.batch_size
        else:
            batch_block = train_num // data.batch_size + 1
        correct = total = total_loss = 0
        random.shuffle(data.train_idx)
        for block in range(batch_block):
            left = block * data.batch_size
            right = left + data.batch_size
            if right > train_num:
                right = train_num
            instance = data.train_idx[left:right]
            batch_word, batch_word_len, word_recover, batch_char, batch_char_len, char_recover, batch_label, mask = generate_batch(
                instance,
                data.use_cuda)
            loss, seq = model.forward(batch_word, batch_word_len, batch_char, batch_char_len, char_recover, batch_label,
                                      mask)
            right_token, total_token = predict_check(seq, batch_label, mask)
            correct += right_token
            total += total_token
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            model.zero_grad()
        epoch_end = time.time()
        print("Epoch:{}. Time:{}. Loss:{}. acc:{}".format(idx, epoch_end - epoch_start, total_loss,
                                                          correct / total))
        evaluate(data, model, "dev",idx)
        evaluate(data, model, "test",idx)
        print("Finish.")


def evaluate(data, model, name,iter):
    if name == "train":
        instances = data.train_idx
        texts = data.train_text
    elif name == "dev":
        instances = data.dev_idx
        texts = data.dev_text
    elif name == "test":
        instances = data.test_idx
        texts = data.test_text
    else:
        raise RuntimeError("Error: no {} evaluate data".format(name))
    model.eval()
    batch_size = data.batch_size
    num = len(instances)
    if num % data.batch_size == 0:
        batch_block = num // data.batch_size
    else:
        batch_block = num // data.batch_size + 1
    correct_num = gold_num = pred_num = 0
    preds = []
    for batch in range(batch_block):
        left = batch * batch_size
        right = left + batch_size
        if right > num:
            right = num
        instance = instances[left:right]
        batch_word, batch_word_len, word_recover, batch_char, batch_char_len, char_recover, batch_label, mask = generate_batch(
            instance,
            data.use_cuda)
        loss, seq = model(batch_word, batch_word_len, batch_char, batch_char_len, char_recover, batch_label, mask)
        gold_list, pred_list = seq_eval(data, seq, batch_label, mask,word_recover)
        # word_recover = word_recover.cpu().data.numpy()
        # gold_list = gold_list[word_recover]
        # pred_list = pred_list[word_recover]
        pred, correct, gold = get_ner_measure(pred_list, gold_list, data.tag_scheme)
        correct_num += correct
        pred_num += pred
        gold_num += gold
        preds.extend(pred_list)
    precision = correct_num / pred_num
    recall = correct_num / gold_num
    f = 2 * precision * recall / (precision + recall)
    print("\t{} Eval: gold_num={}\tpred_num={}\tcorrect_num={}\n\tp:{}\tr:{}\tf:{}".format(name, gold_num, pred_num,
                                                                                           correct_num,
                                                                                           precision, recall, f))
    output_result(texts,preds,data.result_save_dir,name+str(iter))



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
    # print(word_seq_length)
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
    char_seq_tensor = char_seq_tensor[sort_idx].view(-1, max_word_len)
    char_seq_length = char_seq_length[sort_idx].view(-1)

    char_seq_length, char_sort_idx = torch.sort(char_seq_length, descending=True)
    char_seq_tensor = char_seq_tensor[char_sort_idx]
    _, char_seq_recover = char_sort_idx.sort(0, descending=False)
    _, word_seq_recover = sort_idx.sort(0, descending=False)

    return word_seq_tensor, word_seq_length, word_seq_recover, char_seq_tensor, char_seq_length, char_seq_recover, label_seq_tensor, mask


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
