import torch
import argparse
from data import *
import torch.optim as optim
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import time
from seqmodel import SeqModel
from eval import *
from statistic import *

seed_num = 233
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def train(data):
    print("Training model...")
    data.show_data_summary()
    model = SeqModel(data)
    if data.use_cuda:
        model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if data.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=data.lr, momentum=data.momentum, weight_decay=data.weight_decay)
    elif data.optimizer == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=data.lr, weight_decay=data.weight_decay)
    elif data.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=data.lr, weight_decay=data.weight_decay)
    else:
        print("Optimizer Error: {} optimizer is not support.".format(data.optimizer))
    # if data.out_dict:
    #     external_dict(data.out_dict, data.train_file)
    #     external_dict(data.out_dict, data.dev_file)
    #     external = external_dict(data.out_dict, data.test_file)
    #     external_dict(data.out_dict,data.oov_file)
    #     #print(len(external))
    #     #with open('../data/ali_7k/external_dict','w',encoding='utf-8') as fout:
    #     #    for e in external:
    #     #        fout.write(e+'\n')
    for idx in range(data.iter):
        # data.mask_entity(data.iter, idx)
        epoch_start = temp_start = time.time()
        # if data.optimizer == "SGD":
        #    optimizer = lr_decay(optimizer,idx,data.lr_decay,data.lr)
        train_num = len(data.train_text)
        if train_num % data.batch_size == 0:
            batch_block = train_num // data.batch_size
        else:
            batch_block = train_num // data.batch_size + 1
        correct = total = total_loss = 0
        random.shuffle(data.train_idx)
        model.train()
        model.zero_grad()
        for block in range(batch_block):
            left = block * data.batch_size
            right = left + data.batch_size
            if right > train_num:
                right = train_num
            instance = data.train_idx[left:right]
            batch_word, batch_feat, batch_word_len, word_recover, batch_char, batch_char_len, char_recover, batch_label, mask, batch_bert \
                = generate_batch(instance, data.use_cuda, data.use_bert)
            batch_dict = None
            if data.out_dict:
                batch_dict = generate_dict_feature(instance, data, data.use_cuda)
            loss, seq = model.neg_log_likehood(batch_word, batch_feat, batch_word_len, batch_char, batch_char_len,
                                               char_recover, batch_label,
                                               mask, batch_dict, batch_bert)
            right_token, total_token = predict_check(seq, batch_label, mask)
            correct += right_token
            total += total_token
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()
        epoch_end = time.time()
        print("Epoch:{}. Time:{}. Loss:{}. acc:{}".format(idx, epoch_end - epoch_start, total_loss,
                                                          correct / total))
        # torch.save(model,data.model_save_dir+'/model'+str(idx)+'.pkl')
        evaluate(data, model, "dev", idx)
        evaluate(data, model, "test", idx)
        if data.oov_file is not None:
            evaluate(data, model, "oov", idx)
        print("Finish.")


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name, iter):
    if name == "train":
        instances = data.train_idx
    elif name == "dev":
        instances = data.dev_idx
    elif name == "test":
        instances = data.test_idx
    elif name == "oov":
        instances = data.oov_idx
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
        batch_word, batch_feat, batch_word_len, word_recover, batch_char, batch_char_len, char_recover, batch_label, mask, batch_bert \
            = generate_batch(instance, data.use_cuda, data.use_bert)
        batch_dict = None
        if data.out_dict:
            batch_dict = generate_dict_feature(instance, data, data.use_cuda)
        seq = model(batch_word, batch_feat, batch_word_len, batch_char, batch_char_len, char_recover, batch_label, mask,
                    batch_dict, batch_bert)
        gold_list, pred_list = seq_eval(data, seq, batch_label, mask, word_recover)
        # word_recover = word_recover.cpu().data.numpy()
        # gold_list = gold_list[word_recover]
        # pred_list = pred_list[word_recover]
        pred, correct, gold = get_ner_measure(pred_list, gold_list, data.tag_scheme)
        correct_num += correct
        pred_num += pred
        gold_num += gold
        preds.extend(pred_list)
    if pred_num == 0:
        precision = 0
    else:
        precision = correct_num / pred_num
    recall = correct_num / gold_num
    if precision + recall == 0:
        f = 0
    else:
        f = 2 * precision * recall / (precision + recall)
    print("\t{} Eval: gold_num={}\tpred_num={}\tcorrect_num={}\n\tp:{}\tr:{}\tf:{}".format(name, gold_num, pred_num,
                                                                                           correct_num,
                                                                                           precision, recall, f))
    # output_result(texts,preds,data.result_save_dir,name+str(iter))


def predict_check(predict, gold, mask):
    predict = predict.cpu().data.numpy()
    gold = gold.cpu().data.numpy()
    mask = mask.cpu().data.numpy()
    result = (predict == gold)
    right_token = np.sum(result * mask)
    total_token = mask.sum()
    return right_token, total_token


def generate_batch(instance, gpu, use_bert):
    """

    :param instance:  [[word,char,label],...]
    :return:
        zero padding for word and char ,with  their batch length

    """
    batch_size = len(instance)
    words = [ins[0] for ins in instance]
    feats = [np.array(ins[1]) for ins in instance]
    chars = [ins[2] for ins in instance]
    labels = [ins[3] for ins in instance]
    vecs = [ins[4] for ins in instance]
    bert_seq_tensor = None
    word_seq_length = torch.LongTensor(list(map(len, words)))
    feat_num = 0
    if len(feats[0]) > 0:
        feat_num = len(feats[0][0])
    # print(word_seq_length)
    max_seq_length = word_seq_length.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_length), requires_grad=False).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_length), requires_grad=False).long()
    feat_seq_tensor = []
    for idx in range(feat_num):
        feat_seq_tensor.append(torch.zeros((batch_size, max_seq_length), requires_grad=False).long())
    mask = autograd.Variable(torch.zeros(batch_size, max_seq_length)).byte()
    for idx, (seq, label, seq_len) in enumerate(zip(words, labels, word_seq_length)):
        seq_len = seq_len.item()
        word_seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seq_len] = torch.LongTensor(label)
        mask[idx, :seq_len] = torch.Tensor([1] * seq_len)
        for idy in range(feat_num):
            feat_seq_tensor[idy][idx, :seq_len] = torch.LongTensor(feats[idx][:, idy])
    word_seq_length, sort_idx = torch.sort(word_seq_length, descending=True)
    word_seq_tensor = word_seq_tensor[sort_idx]
    label_seq_tensor = label_seq_tensor[sort_idx]
    for idx in range(feat_num):
        feat_seq_tensor[idx] = feat_seq_tensor[idx][sort_idx]
    mask = mask[sort_idx]

    # char
    pad_chars = [chars[idx] + [[0]] * (max_seq_length - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_length, max_word_len), requires_grad=False).long()
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
    if use_bert:
        bert_seq_tensor = pad_sequence(vecs, batch_first=True)
        bert_seq_tensor = bert_seq_tensor[sort_idx]
        if gpu:
            bert_seq_tensor = bert_seq_tensor.cuda()

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feat_num):
            feat_seq_tensor[idx] = feat_seq_tensor[idx].cuda()
        word_seq_length = word_seq_length.cuda()
        word_seq_recover = word_seq_recover.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_length = char_seq_length.cuda()
        char_seq_recover = char_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()

    return word_seq_tensor, feat_seq_tensor, word_seq_length, word_seq_recover, \
           char_seq_tensor, char_seq_length, char_seq_recover, label_seq_tensor, mask, bert_seq_tensor


def generate_dict_feature(instance, data, gpu):
    batch_size = len(instance)
    words = [ins[0] for ins in instance]
    word_seq_length = torch.LongTensor(list(map(len, words)))
    max_seq_length = word_seq_length.max()
    word_seq_tensor = autograd.Variable(torch.zeros(batch_size, max_seq_length, 8), requires_grad=False).float()
    dict_features = []
    for seq in words:
        dict_feature = []
        for idx, s in enumerate(seq):
            word_feat = []
            for idy in range(8):
                ngram = ""
                if idy % 2 == 0:
                    for idz in range(idx - idy // 2 - 1, idx + 1):
                        ngram += data.word_alphabet.get_instance(seq[idz])
                else:
                    for idz in range(idx, idx + idy // 2 + 2):
                        if idz >= len(seq):
                            idz = len(seq) - idz - 1
                        ngram += data.word_alphabet.get_instance(seq[idz])
                word_feat.append(compute_similarity(ngram, data.out_dict))
            dict_feature.append(soft_feat(word_feat))
        dict_features.append(dict_feature)
    for idx, seq in enumerate(dict_features):
        for idy, word in enumerate(seq):
            word_seq_tensor[idx, idy, :] = torch.FloatTensor(seq[idy])
    if data.use_cuda:
        word_seq_tensor = word_seq_tensor.cuda()
    return word_seq_tensor


def compute_similarity(word, dict):
    sim_val = 0
    if word in dict:
        sim_val += int(dict[word])
    else:
        for k, v in dict.items():
            if word in k:
                sim_val += (len(word) / len(k)) * int(dict[k])
    if sim_val == 0:
        sim_val -= len(word)
    return sim_val


def soft_feat(feat_list):
    sum = 0
    for f in feat_list:
        sum += f
    return [f / sum if sum != 0 else 0 for f in feat_list]


def tag_with_model(data):
    model = torch.load(data.model_path)
    print("load model finish.....")
    model.eval()
    instances = data.test_idx
    texts = data.test_text
    batch_size = data.batch_size
    num = len(instances)
    if num % data.batch_size == 0:
        batch_block = num // data.batch_size
    else:
        batch_block = num // data.batch_size + 1
    fout = open('model_decode', 'w', encoding='utf-8')
    for batch in range(batch_block):
        left = batch * batch_size
        right = left + batch_size
        if right > num:
            right = num
        instance = instances[left:right]
        batch_word, batch_word_len, word_recover, batch_char, batch_char_len, char_recover, batch_label, mask, batch_bert = generate_batch(
            instance,
            data.use_cuda, data.use_bert)
        loss, seq = model(batch_word, batch_word_len, batch_char, batch_char_len, char_recover, batch_label, mask,
                          batch_bert)
        gold_list, pred_list = seq_eval(data, seq, batch_label, mask, word_recover)
        text = texts[left:right]
        for idx, sen in enumerate(text):
            for idy, s in enumerate(sen[0]):
                fout.write(s + '\t' + pred_list[idx][idy] + '\n')
            fout.write('\n')
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL NER")
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
        data.get_instance("oov")
        data.build_alphabet()
        data.get_instance_index("train")
        data.get_instance_index("dev")
        data.get_instance_index("test")
        data.get_instance_index("oov")
        print(data.word_alphabet.size(), data.char_alphabet.size(), data.label_alphabet.size())
        if data.pretrain:
            data.build_pretrain_emb()
        data.word_alphabet_conll('word_conll')
        train(data)
        print(data.word_alphabet.size(), data.char_alphabet.size(), data.label_alphabet.size())
    elif status.lower() == "tag":
        data.get_instance("train")
        data.build_alphabet()
        data.get_instance("test")
        data.get_instance_index("test")
        tag_with_model(data)
