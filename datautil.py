UNK = '-unknown'
PAD = '-padding-'
NULL = 'NULL'


class Instance:
    def __init__(self):
        self.word = []
        self.label = []


class Code:
    def __init__(self):
        self.feature_code = []
        self.label_code = []


class CoNLL2000(Instance):
    def __init__(self, path):
        super(Instance, self).__init__()
        self.result = self.load_data(path)
        self.sentence = len(self.result)
        self.token = self.cal_token(self.result)

    @staticmethod
    def load_data(path):
        result = []
        with open(path, 'r', encoding='utf8') as fin:
            inst = Instance()
            for line in fin:
                if line != '\n':
                    word_info = line.strip().split(" ")
                    inst.word.append(word_info[0])
                    inst.label.append(word_info[2])
                else:
                    result.append(inst)
                    inst = Instance()
        return result

    @staticmethod
    def cal_token(result):
        num = 0
        for r in result:
            num += len(r.word)
        return num


class Vocab:
    def __init__(self, lower=False):
        self.id2word = {}
        self.word2id = {}
        self.id2label = {}
        self.label2id = {}
        self.lower = lower

    def create_vocab(self, result):
        for r in result:
            for i, w in enumerate(r.word):
                label = r.label[i]
                if self.lower:
                    w = w.lower()
                if w not in self.word2id:
                    idx = len(self.id2word)
                    self.word2id[w] = idx
                    self.id2word[idx] = w
                if label not in self.label2id:
                    idx = len(self.label2id)
                    self.label2id[label] = idx
                    self.id2label[idx] = label
        vlen = len(self.id2word)
        self.word2id[UNK] = vlen
        self.id2word[vlen] = UNK
        self.word2id[PAD] = vlen + 1
        self.id2word[vlen + 1] = PAD
        vlen = len(self.id2label)
        self.label2id[NULL] = vlen
        self.id2label[vlen] = NULL


class FeatureEncoder:
    @staticmethod
    def encode(result, feature_vocab, label_vocab):
        features = []
        for r in result:
            code = Code()
            for idx, w in enumerate(r.word):
                label = r.label[idx]
                if w in feature_vocab:
                    code.feature_code.append(feature_vocab[w])
                else:
                    code.feature_code.append(feature_vocab[UNK])
                if label in label_vocab:
                    code.label_code.append(label_vocab[r.label[idx]])
                else:
                    code.label_code.append(label_vocab[NULL])
            features.append(code)
        return features
