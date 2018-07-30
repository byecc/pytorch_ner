def seq_eval(data, pred, gold):
    pred_list = gold_list = []
    seq_len = gold.size(1)
    pred_tag = pred.cpu().data.numpy()
    gold_tag = gold.cpu().data.numpy()
    for idx in range(data.batch_size):
        pred = [data.label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if pred_tag[idx][idy] != 0]
        gold = [data.label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if gold_tag[idx][idy] != 0]
        pred_list.append(pred)
        gold_list.append(gold)
    return gold_list,pred_list

def get_ner_measure(pred,gold,scheme):
    sen_num = len(pred)
    predict_num = correct_num = gold_num = 0
    for idx in range(sen_num):
        if scheme == "BIO":
            gold_entity = get_entity(gold)
            pred_entity = get_entity(pred)
            predict_num += len(pred_entity)
            gold_num += len(gold_entity)
            correct_num += len(list(set(gold_entity).intersection(set(pred_entity))))
        else: # "BMES"
            pass
    return predict_num,correct_num,gold_num


def get_entity(label_list):
    sen_len = len(label_list)
    entity_list = []
    entity = None
    entity_index =None
    for idx in range(sen_len):
        current = label_list[idx]
        if "B-" in current:
            if entity is not None:
                entity_list.append("["+entity_index+"]"+entity)
                entity = None
                entity_index = None
            else:
                entity = current.split("-")[1]
                entity_index = str(idx)
        elif "I-" in current:
            if entity is not None:
                entity_index += str(idx)
            else:
                continue
        elif "O" in current:
            continue
        else:
            print("Label Error.")
    return entity_list


class Eval:
    def __init__(self, data, pred, gold):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def acc(self):
        return self.correct_num / self.gold_num

    def getFscore(self):
        pass
