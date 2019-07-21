## Basic two layer NN for sam surface (input vec is all positions)

from scratch.neural_net.lib import *

from model import ConvNet
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

## Params ##
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

## Set up data 
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))


## Initialize net, loss function, and optimizer
net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


## Train ##

# batches per epoch
n_batches = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Forward pass
        net_out = net(images)
        loss = criterion(net_out, labels)
        loss_list.append(loss.item())

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get accuracy
        total = labels.size(0)
        _, predicted = torch.max(net_out.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i % 100 == 0):
            print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".
                  format(epoch, num_epochs, i, n_batches, loss.item(), correct/total * 100))

