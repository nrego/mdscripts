## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.sam.util import *
from scratch.neural_net.util import *
from scratch.neural_net.mnist_net import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

do_cnn = True
no_run = False

feat_vec, energies, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep()
pos = pos_ext[patch_indices]

data_partition_gen = partition_data(feat_vec, energies, n_groups=7, batch_size=120, do_cnn=do_cnn)
loader = init_data_and_loaders(feat_vec, energies, 200, False, do_cnn)
dataset = loader.dataset
mses = []

for i_round, (train_loader, test_loader) in enumerate(data_partition_gen):

    if no_run:
        break

    if do_cnn:
        net = SAMConvNet(n_channels=12, kernel_size=5, n_hidden=32)
    else:
        net = SAM2LNet(n_hidden1=12, n_hidden2=12)
    # minimize MSE of predicted energies
    criterion = nn.MSELoss()    

    print("Training/validation round {}".format(i_round))
    print("============================")
    print("\n")

    print("...Training")
    losses = train(net, criterion, train_loader, test_loader,do_cnn, learning_rate=0.001, weight_decay=0.0, epochs=3000)
    print("    DONE...")
    print("\n")
    print("...Testing round {}".format(i_round))
    test_X, test_y = iter(test_loader).next()
    # So there are no shenanigans with MSE
    if not do_cnn:
        test_X = test_X.view(-1, 8*8)

    pred = net(test_X).detach()
    mse = criterion(pred, test_y).item()
    print("Validation MSE: {:.2f}".format(mse))

    mses.append(mse)

    if i_round == 0:
        break
    del net, criterion
