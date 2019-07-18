import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from IPython import embed


# Runs basic linear regression on our number of edges
#   For testing 
class TestSAMNet(nn.Module):
    def __init__(self, n_dim=2):
        super(TestSAMNet, self).__init__()
        self.fc1 = nn.Linear(n_dim, 1)

    def forward(self, x):
        x = self.fc1(x)

        return x

    @property
    def coef_(self):
        return self.fc1.weight.detach().numpy()

    @property
    def intercept_(self):
        return self.fc1.bias.item()

# One hidden layer net
class SAMNet(nn.Module):
    def __init__(self, n_patch_dim=36, n_hidden=36, n_layers=1, n_out=1, drop_out=0.0):
        super(SAMNet, self).__init__()

        layers = []

        for i in range(n_layers):
            if i == 0:
                L = nn.Linear(n_patch_dim, n_hidden)
            else:
                L = nn.Linear(n_hidden, n_hidden)
            
            layers.append(L)
            layers.append(nn.ReLU())
            
            if drop_out > 0:
                layers.append(nn.Dropout(drop_out))

        self.layers = nn.Sequential(*layers)
        self.o = nn.Linear(n_hidden, n_out)

    def forward(self, x):

        out = self.layers(x)
        out = self.o(out)

        return out

    @property
    def n_layers(self):
        return len(self.layers)
