from pos.config import *
from datautil import *
from pos.train import *

args = parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)

if __name__ == "__main__":
    train_Conll = CoNLL2000(args.train_path)
    test_Conll = CoNLL2000(args.test_path)
    vocab = Vocab()
    vocab.create_vocab(train_Conll.result)
    args.feat_padidx = len(vocab.word2id) - 1
    args.label_padidx = len(vocab.label2id) - 1
    train_data = FeatureEncoder.encode(train_Conll.result, vocab.word2id, vocab.label2id)
    test_data = FeatureEncoder.encode(test_Conll.result, vocab.word2id, vocab.label2id)
    args.embed_num = len(vocab.word2id)
    args.label_num = len(vocab.label2id)
    trainer = Trainer(args)
    print('——————————start——————————')
    print(args)
    args.vocab = vocab.word2id
    trainer.postag_lstm_train(train_data, test_data)
    print('---finish---')
