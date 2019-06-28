import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from scratch.neural_net.mnist_net import *
from torchvision import datasets, transforms

import matplotlib as mpl
from matplotlib import pyplot as plt
from IPython import embed


num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.13066,), (0.30811,))])
dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataset_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
