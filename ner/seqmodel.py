import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqModel(nn.Module):
    def __init__(self,data):
        super(SeqModel, self).__init__()
        self.args = data

