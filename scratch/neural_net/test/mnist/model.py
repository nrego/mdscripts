## Basic two layer NN for sam surface (input vec is all positions)

from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from IPython import embed

import argparse

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Input: 1 x 28 x 28
        # Conv Output: 4 x 28 x 28
        # Pool Output: 4 x 14 x 14
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Input: 32 x 14 x 14
        # Conv Output: 64 x 14 x 14
        # Pool Output: 64 x 7 x 7
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()

        ## Fully connected layers
        self.fc1 = nn.Linear(8 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        # ravel channels and grids to (64*7*7)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out