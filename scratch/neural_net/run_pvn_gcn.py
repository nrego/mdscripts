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

xvals = np.arange(0,124)
xvals = np.array([xvals**4,
                  xvals**3,
                  xvals**2,
                  xvals**1,
                  xvals**0])
xvals = torch.tensor(xvals.astype(np.float32))

def loss_poly(net_out, target, criterion, p_min, p_range_mat):

    p_min_mat = np.ones((net_out.shape[0], p_min.size), dtype=np.float32)
    p_min_mat *= p_min
    p_min_mat = torch.tensor(p_min_mat)

    pred = torch.matmul(net_out, p_range_mat) + p_min_mat
    pred = torch.matmul(pred, xvals)

    act = torch.matmul(target, p_range_mat) + p_min_mat
    act = torch.matmul(act, xvals)
    loss = criterion(pred, act)

    return loss

def norm_poly(poly):
    scaled_poly = poly.copy()
    for i in range(poly.shape[1]):
        scaled_poly[:, i] = (poly[:,i] - poly[:,i].min())/(poly[:,i].max()-poly[:,i].min())

def inflate(poly, scaled_poly):
    inflate = scaled_poly.copy()
    for i in range(poly.shape[1]):
        inflate[:,i] = scaled_poly[:,i]*(poly[:,i].max()-poly[:,i].min()) + poly[:,i].min()

    return inflate


no_run = False


feat_vec, energies, poly, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep()


p_min = poly.min(axis=0).astype(np.float32)
p_max = poly.max(axis=0).astype(np.float32)
p_range = p_max - p_min
scaled_poly = (poly - p_min) / (p_range)
y_vec = scaled_poly

p_range_mat = torch.tensor(np.diag(p_range))



pos = pos_ext[patch_indices]
adj_mat = adj_mat.astype(np.float32)
node_deg = np.diag(adj_mat.sum(axis=0))
mask = node_deg > 0
d_inv = node_deg.copy()
d_inv[mask] = node_deg[mask]**-1


data_partition_gen = list(partition_data(feat_vec, y_vec, n_groups=5, batch_size=200, do_cnn=do_cnn))
loader = init_data_and_loaders(feat_vec, y_vec, 200, False, do_cnn)
dataset = loader.dataset
mses = []

## Round 1 - train on normalized
for i_round, (train_loader, test_loader) in enumerate(data_partition_gen):

    if no_run:
        break

    outdim = 1 if y_vec.ndim == 1 else y_vec.shape[1]

    net = SAMGraphNet3L(adj_mat, n_hidden=18, n_out=outdim)
    #net = SAMGraphNet(adj_mat, n_hidden=36, n_out=outdim)
    # minimize MSE of predicted energies
    criterion = nn.MSELoss()    
    #criterion = nn.NLLLoss()

    print("Training/validation round {}".format(i_round))
    print("============================")
    print("\n")

    print("...Training")
    losses = train(net, criterion, train_loader, test_loader, do_cnn, learning_rate=0.001, weight_decay=0.0, epochs=1000)
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

    #mses.append(mse)

    if i_round == 0:
        break

save_net(net)

#data_partition_gen = partition_data(feat_vec, y_vec, n_groups=5, batch_size=200, do_cnn=do_cnn)
## Round 2 - train on mse between energies

for i_round, (train_loader, test_loader) in enumerate(data_partition_gen):
    net = load_net()
    if no_run:
        break

    outdim = 1 if y_vec.ndim == 1 else y_vec.shape[1]

    #net = SAMGraphNet3L(adj_mat, n_hidden=64, n_out=outdim)
    # minimize MSE of predicted energies
    criterion = nn.MSELoss()    
    #criterion = nn.NLLLoss()

    print("Training/validation round {} of {}".format(i_round+1, len(data_partition_gen)))
    print("============================")
    print("\n")

    print("...Training")
    losses = train(net, criterion, train_loader, test_loader, do_cnn, learning_rate=0.001, weight_decay=0.0, epochs=6000, break_out=3, loss_fn=loss_poly, loss_fn_args=(p_min, p_range_mat))
    print("    DONE...")
    print("\n")
    print("...Testing round {}".format(i_round))
    test_X, test_y = iter(test_loader).next()
    # So there are no shenanigans with MSE
    test_X = test_X.view(-1, test_X.shape[-1])
    test_y = test_y.view(-1, test_y.shape[-1])

    pred = net(test_X).detach()
    mse = loss_poly(pred, test_y, criterion, p_min, p_range_mat).item()
    print("Validation MSE: {:.2f}".format(mse))

    mses.append(mse)

    if i_round == len(data_partition_gen) - 1:
        break
    del net, criterion

#pred = net(torch.tensor(feat_vec)).detach()
p_min_mat = np.ones((pred.shape[0], 5))
p_min_mat *= p_min
p_min_mat = torch.tensor(p_min_mat.astype(np.float32))
pred = torch.matmul(pred, p_range_mat)
pred = pred + p_min_mat
e_pred = torch.matmul(pred, xvals).numpy()

act = torch.matmul(test_y, p_range_mat)
act = act + p_min_mat
e_act = np.dot(act, xvals.numpy())

