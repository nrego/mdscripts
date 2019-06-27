import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from scratch.neural_net.mnist_net import BasicNet
from torchvision import datasets, transforms

import matplotlib as mpl
from matplotlib import pyplot
from IPython import embed

## driver ##

def init_data_and_loaders(batch_size=200):
    dataset_train = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ]))

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    return train_loader


def train(net, criterion, train_loader, learning_rate=0.01, epochs=10, log_interval=10):

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Total number of data points
    n_data = len(train_loader.dataset)
    # num training rounds per epoch (number data points / batch size)
    n_rounds = len(train_loader)
    batch_size = int(n_data / n_rounds)
    # total number of training steps
    n_steps = epochs * n_rounds
    losses = np.zeros(n_steps)
    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            #embed()
            optimizer.zero_grad()
            net_out = net(data)
            
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    return losses



def validate(net, dataset_test, criterion):

    loader = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test))

    data, labels = next(iter(loader))
    data = data.view(-1, 28*28)
    net_out = net(data).detach()

    test_loss = criterion(net_out, labels).item()
    # Prediction is index of max of log probability
    pred = net_out.max(1)[1]
    correct = pred.eq(labels).sum()

    print('\nTest set: avg loss {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss/len(dataset_test), correct, len(dataset_test), 100.*correct/len(dataset_test)))



net = BasicNet()
# Loss function
criterion = nn.NLLLoss()
train_loader = init_data_and_loaders()
dataset = train_loader.dataset


losses = train(net, criterion, train_loader)

dataset_test = datasets.MNIST('../data', train=False, download=True, transform=train_loader.dataset.transform)

validate(net, dataset_test, criterion)


