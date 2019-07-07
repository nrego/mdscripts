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


no_run = False

feat_vec, energies, poly_5, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep()
pos = pos_ext[patch_indices]
adj_mat = adj_mat.astype(np.float32)
node_deg = np.diag(adj_mat.sum(axis=0))
mask = node_deg > 0
d_inv = node_deg.copy()
d_inv[mask] = node_deg[mask]**-1

data_partition_gen = partition_data(feat_vec, energies, n_groups=5, batch_size=200, do_cnn=do_cnn)
loader = init_data_and_loaders(feat_vec, energies, 200, False, do_cnn)
dataset = loader.dataset
mses = []

for i_round, (train_loader, test_loader) in enumerate(data_partition_gen):

    if no_run:
        break

    net = SAMGraphNet(adj_mat, n_hidden1=64)
    # minimize MSE of predicted energies
    criterion = nn.MSELoss()    

    print("Training/validation round {}".format(i_round))
    print("============================")
    print("\n")

    print("...Training")
    losses = train(net, criterion, train_loader, test_loader, do_cnn, learning_rate=0.005, weight_decay=0.0, break_out=10, epochs=3000)
    print("    DONE...")
    print("\n")
    print("...Testing round {}".format(i_round))
    test_X, test_y = iter(test_loader).next()
    # So there are no shenanigans with MSE
    test_X = test_X.view(-1, test_X.shape[-1])
    test_y = test_y.view(-1, test_y.shape[-1])

    pred = net(test_X).detach()
    mse = criterion(pred, test_y).item()
    print("Validation MSE: {:.2f}".format(mse))

    mses.append(mse)

    if i_round == 0:
        break
    del net, criterion