class Eval:
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def acc(self):
        return self.correct_num/self.gold_num

    def getFscore(self):
        pass
