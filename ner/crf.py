import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


START_TAG=-2
STOP_TAG=-1

class CRF(nn.Module):
    def __init__(self,target_size,gpu):
        super(CRF,self).__init__()
        self.target_size = target_size
        self.gpu = gpu
        transition_mat = torch.zeros(self.target_size+2,self.target_size+2)


