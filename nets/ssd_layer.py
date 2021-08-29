from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
# from utils.box_utils import decode, nms
# from utils.config import Config


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gama = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.tensor(self.n_channels))
        self.reset_parameters()

    def resrt_parameters(self):
        init.constant_(self.weight, self.gama)

    def forward(self,x):
        norm=x.pow(2).sum(dim=1,keepdim=True).sqrt()+self.eps
        # x/norm
        x=torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
