import numpy as np
import sys
import pickle
import os

START = "</s>"
PAD = "</pad>"
UNKNOWN = "</unk>"


class Alphabet:
    def __init__(self, name, unknown_label=True):
        self.name = name
        self.instance2index = {}
        self.instances = []
        # self.keep_growing = keep_growing
        self.index = 1
        if unknown_label:
            self.add(UNKNOWN)

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.index
            self.index += 1

    def size(self):
        return len(self.instances) + 1

    def get_instance(self, index):
        return self.instances[index - 1]


class Data:
    def __init__(self):
        self.config = {}
        self.train_text = []
        self.dev_text = []
        self.test_text = []
        self.train_idx = []
        self.dev_idx = []
        self.test_idx = []

        self.word_alphabet = Alphabet("word")
        self.char_alphabet = Alphabet("char")
        self.label_alphabet = Alphabet("label", unknown_label=False)
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0

        self.status = None
        self.train_file = None
        self.dev_file = None
        self.test_file = None
        self.pretrain_file = None
        self.word_embed_path = None
        self.word_embed_save = None
        self.char_embed_path = None
        self.model_save_dir = None
        self.dataset = None
        self.word_normalize = False
        self.word_feature_extractor = "LSTM"
        self.use_char = False
        self.char_feature_extractor = "LSTM"
        self.use_crf = False
        self.use_cuda = False
        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.tag_scheme = None

        # hyperparameters
        self.pretrain = True
        self.word_embed_dim = None
        self.char_embed_dim = None
        self.word_seq_feature = None
        self.char_seq_feature = None
        self.optimizer = None
        self.hidden_dim = None
        self.char_hidden_dim = None
        self.bilstm = None
        self.lstm_layer = None
        self.batch_size = None
        self.dropout = None
        self.lr = None
        self.momentum = None
        self.weight_decay = None
        self.iter = None
        self.fine_tune = False
        self.attention = False
        self.attention_dim = None
        self.average_loss = False
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.number_normalized = False

    def get_instance(self, name):
        if name == 'train':
            self.read_data(self.train_file, name, self.dataset)
        elif name == 'dev':
            self.read_data(self.dev_file, name, self.dataset)
        elif name == 'test':
            self.read_data(self.test_file, name, self.dataset)
        else:
            print('Get Instance Error: pls set correct data instance.')

    def read_config(self, file):
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                if len(line) > 0 and line[0] == '#':
                    continue
                if '=' in line:
                    parameter = line.strip().split('=')
                    if parameter[0] not in self.config:
                        self.config[parameter[0]] = parameter[1]
                    else:
                        print('Warning : duplicate parameter found.')
        item = 'train_file'
        if item in self.config:
            self.train_file = self.config[item]
        item = 'dev_file'
        if item in self.config:
            self.dev_file = self.config[item]
        item = 'test_file'
        if item in self.config:
            self.test_file = self.config[item]
        item = 'dataset'
        if item in self.config:
            self.dataset = self.config[item]
        item = 'status'
        if item in self.config:
            self.status = self.config[item]
        item = 'word_feature_extractor'
        if item in self.config:
            self.word_feature_extractor = self.config[item]
        item = 'use_char'
        if item in self.config:
            self.use_char = str2bool(self.config[item])
        item = 'char_feature_extractor'
        if item in self.config:
            self.char_feature_extractor = self.config[item]
        item = 'use_crf'
        if item in self.config:
            self.use_crf = str2bool(self.config[item])
        item = 'word_embed_path'
        if item in self.config:
            self.word_embed_path = self.config[item]
        item = 'char_embed_path'
        if item in self.config:
            self.char_embed_path = self.config[item]
        item = 'word_embed_dim'
        if item in self.config:
            self.word_embed_dim = int(self.config[item])
        item = 'char_embed_dim'
        if item in self.config:
            self.char_embed_dim = int(self.config[item])
        item = 'hidden_dim'
        if item in self.config:
            self.hidden_dim = int(self.config[item])
        item = 'char_hidden_dim'
        if item in self.config:
            self.char_hidden_dim = int(self.config[item])
        item = 'use_cuda'
        if item in self.config:
            self.use_cuda = str2bool(self.config[item])
        item = 'lr'
        if item in self.config:
            self.lr = float(self.config[item])
        item = 'momentum'
        if item in self.config:
            self.momentum = float(self.config[item])
        item = 'weight_decay'
        if item in self.config:
            self.weight_decay = float(self.config[item])
        item = 'dropout'
        if item in self.config:
            self.dropout = float(self.config[item])
        item = 'iter'
        if item in self.config:
            self.iter = int(self.config[item])
        item = 'optimizer'
        if item in self.config:
            self.optimizer = self.config[item]
        item = 'batch_size'
        if item in self.config:
            self.batch_size = int(self.config[item])
        item = 'lstm_layer'
        if item in self.config:
            self.lstm_layer = int(self.config[item])
        item = 'fine_tune'
        if item in self.config:
            self.fine_tune = str2bool(self.config[item])
        item = 'attention'
        if item in self.config:
            self.attention = str2bool(self.config[item])
        item = 'attention_dim'
        if item in self.config:
            self.attention_dim = int(self.config[item])
        item = 'bilstm'
        if item in self.config:
            self.bilstm = str2bool(self.config[item])
        item = 'average_loss'
        if item in self.config:
            self.average_loss = str2bool(self.config[item])
        item = 'norm_word_emb'
        if item in self.config:
            self.norm_word_emb = str2bool(self.config[item])
        item = 'norm_char_emb'
        if item in self.config:
            self.norm_char_emb = str2bool(self.config[item])
        item = 'tag_scheme'
        if item in self.config:
            self.tag_scheme = self.config[item]
        item = 'word_embed_save'
        if item in self.config:
            self.word_embed_save = self.config[item]
        item = 'number_normalized'
        if item in self.config:
            self.number_normalized = str2bool(self.config[item])

    def show_config(self):
        for k, v in self.config.items():
            print(k, v)

    def read_data(self, path, data, datasetname):
        samples = []
        with open(path, 'r', encoding='utf-8') as fin:
            if datasetname == 'conll2003':
                # fin.readline()
                word_instances = []
                char_instances = []
                label_instances = []
                for line in fin:
                    if line != '\n':
                        info = line.strip().split()
                        word = info[0]
                        ner_tag = info[3]
                        word_instances.append(word)
                        char_instances.append(list(word))
                        label_instances.append(ner_tag)
                    else:
                        samples.append([word_instances, char_instances, label_instances])
                        word_instances = []
                        char_instances = []
                        label_instances = []
                samples.append([word_instances, char_instances, label_instances])
        if data == "train":
            self.train_text = samples
        elif data == "dev":
            self.dev_text = samples
        elif data == "test":
            self.test_text = samples
        else:
            print("Data Error:pls set train/dev/test data parameter.")

    def build_alphabet(self):
        for sample in self.train_text:
            for word in sample[0]:
                self.word_alphabet.add(word)
            for char in sample[1][0]:
                self.char_alphabet.add(char)
            for label in sample[2]:
                self.label_alphabet.add(label)
        for sample in self.dev_text:
            for word in sample[0]:
                self.word_alphabet.add(word)
            for char in sample[1][0]:
                self.char_alphabet.add(char)
            for label in sample[2]:
                self.label_alphabet.add(label)
        for sample in self.test_text:
            for word in sample[0]:
                self.word_alphabet.add(word)
            for char in sample[1][0]:
                self.char_alphabet.add(char)
            for label in sample[2]:
                self.label_alphabet.add(label)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        print("build alphabet finish.")

    def get_instance_index(self, data):
        instance_idx = []
        if data == "train":
            samples = self.train_text
        elif data == "dev":
            samples = self.dev_text
        elif data == "test":
            samples = self.test_text
        for sample in samples:
            word_idx, char_idx, label_idx = [], [], []
            for word in sample[0]:
                if word in self.word_alphabet.instance2index:
                    word_idx.append(self.word_alphabet.instance2index[word])
                else:
                    word_idx.append(self.word_alphabet.instance2index[UNKNOWN])
            for char in sample[1]:
                chars = []
                for c in char:
                    if c in self.char_alphabet.instance2index:
                        chars.append(self.char_alphabet.instance2index[c])
                    else:
                        chars.append(self.char_alphabet.instance2index[UNKNOWN])
                char_idx.append(chars)
            for label in sample[2]:
                if label in self.label_alphabet.instance2index:
                    label_idx.append(self.label_alphabet.instance2index[label])
                else:
                    label_idx.append(self.label_alphabet.instance2index[UNKNOWN])
            instance_idx.append([word_idx, char_idx, label_idx])
        if data == "train":
            self.train_idx = instance_idx
        elif data == "dev":
            self.dev_idx = instance_idx
        elif data == "test":
            self.test_idx = instance_idx
        else:
            print("Data Error:pls set train/dev/test data parameter.")

    def build_pretrain_emb(self):
        if self.word_embed_path:
            self.pretrain_word_embedding, self.word_embed_dim = build_pretrain_embedding(self.word_embed_save,
                                                                                         self.word_embed_path,
                                                                                         self.word_alphabet,
                                                                                         self.word_embed_dim,
                                                                                         self.norm_word_emb)
        elif self.char_embed_path:
            self.pretrain_char_embedding, self.char_embed_dim = build_pretrain_embedding(self.char_embed_path,
                                                                                         self.char_embed_path,
                                                                                         self.char_alphabet,
                                                                                         self.char_embed_dim,
                                                                                         self.norm_char_emb)


def build_pretrain_embedding(embedding_save, embedding_path, word_alphabet, emb_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_save is not None and os.path.exists(embedding_save):
        pretrain_emb = pickle.load(open(embedding_save, 'rb'))
        embedd_dim = pretrain_emb.shape[1]
        return pretrain_emb,embedd_dim
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    pickle.dump(pretrain_emb, open(embedding_save, 'wb'))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as file:
        file.readline()
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 != len(tokens):
                    continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False
