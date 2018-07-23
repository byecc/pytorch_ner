START = "</s>"
PAD = "</pad>"
UNKNOWN = "</unk>"


class Alphabet:
    def __init__(self, name):
        self.name = name
        self.instance2index = {}
        self.instances = []
        # self.keep_growing = keep_growing
        self.index = 0
        self.add(UNKNOWN)

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.index
            self.index += 1


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
        self.label_alphabet = Alphabet("label")
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0

        self.status = None
        self.train_file = None
        self.dev_file = None
        self.test_file = None
        self.word_embed_path = None
        self.model_save_dir = None
        self.dataset = None
        self.word_normalize = False
        self.word_feature_extractor = "LSTM"
        self.use_char = False
        self.char_feature_extractor = "LSTM"
        self.use_crf = False

        # hyperparameters
        self.word_embed_dim = None
        self.char_embed_dim = None
        self.word_seq_feature = None
        self.char_seq_feature = None
        self.optimizer = None
        self.hidden_dim = None
        self.char_hidden_dim = None
        self.bilstm = None
        self.lstm_layer = None
        self.dropout = None
        self.lr = None

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
            self.word_feature_extractor=self.config[item]
        item = 'use_char'
        if item in self.config:
            self.use_char = self.config[item]
        item = 'char_feature_extractor'
        if item in self.config:
            self.char_feature_extractor=self.config[item]
        item = 'use_crf'
        if item in self.config:
            self.use_crf =  self.config[item]
        item='word_embed_path'
        if item in self.config:
            self.word_embed_path = self.config[item]
        item = 'word_embed_dim'
        if item in self.config:
            self.word_embed_dim = self.config[item]
        item = 'char_embed_dim'
        if item in self.config:
            self.char_embed_dim = self.config[item]
        item = 'hidden_dim'
        if item in self.config:
            self.hidden_dim = self.config[item]
        item = 'char_hidden_dim'
        if item in self.config:
            self.char_hidden_dim = self.config[item]



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
            print(self.train_text[-1])
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
        self.word_alphabet_size = len(self.word_alphabet.instance2index)
        self.char_alphabet_size = len(self.char_alphabet.instance2index)
        self.label_alphabet_size = len(self.label_alphabet.instance2index)
        print("build alphabet finish.")

    def get_instance_index(self, data):
        word_idx, char_idx, label_idx = [], [], []
        for sample in self.train_text:
            for word in sample[0]:
                if word in self.word_alphabet:
                    word_idx.append(self.word_alphabet.instance2index[word])
                else:
                    word_idx.append(self.word_alphabet.instance2index[UNKNOWN])
            for char in sample[1][0]:
                if char in self.char_alphabet:
                    char_idx.append(self.char_alphabet.instance2index[char])
                else:
                    char_idx.append(self.char_alphabet.instance2index[UNKNOWN])
            for label in sample[2]:
                if label in self.label_alphabet:
                    label_idx.append(self.label_alphabet.instance2index[label])
                else:
                    label_idx.append(self.label_alphabet.instance2index[UNKNOWN])
        if data == "train":
            self.train_idx = [word_idx, char_idx, label_idx]
        elif data == "dev":
            self.dev_idx = [word_idx, char_idx, label_idx]
        elif data == "test":
            self.test_idx = [word_idx, char_idx, label_idx]
        else:
            print("Data Error:pls set train/dev/test data parameter.")
