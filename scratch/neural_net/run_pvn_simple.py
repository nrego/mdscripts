## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import argparse

xvals = np.arange(0,124)
xvals = np.array([xvals**4,
                  xvals**3,
                  xvals**2,
                  xvals**1,
                  xvals**0])
xvals = torch.tensor(xvals.astype(np.float32))


## Acts on torch tensors
#    Transform output vectors (normalized polynomial coefs)
#      To 
def loss_poly(net_out, target, criterion, p_min, p_range_mat, xvals):

    # Depends on number of datapoints (batch size), so has to be setup here
    p_min_mat = np.ones((net_out.shape[0], p_min.size), dtype=np.float32)
    p_min_mat *= p_min
    p_min_mat = torch.tensor(p_min_mat)

    pred = torch.matmul(net_out, p_range_mat) + p_min_mat
    pred = torch.matmul(pred, xvals)

    act = torch.matmul(target, p_range_mat) + p_min_mat
    act = torch.matmul(act, xvals)

    loss = criterion(pred, act)

    return loss


def normalize(poly, y_min, y_max):
    return (poly - y_min) / (y_max - y_min)

def unnormalize(normed_poly, y_min, y_max):
    return (normed_poly) * (y_max-y_min) + y_min


def run(args):

    # Load input features
    feat_vec, energies, poly, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep(args.datainput)

    p_min = poly.min(axis=0).astype(np.float32)
    p_max = poly.max(axis=0).astype(np.float32)
    p_range = p_max - p_min

    # Position of patch indices (only different if we're using an extended grid)
    pos = pos_ext[patch_indices]

    data_partition_gen = partition_data(feat_vec, poly, n_groups=args.n_valid, batch_size=args.batch_size)


    ## Round 1 - train on normalized coefficients
    for i_round, (train_X, train_y, test_X, test_y) in enumerate(data_partition_gen):

        # Setup data loaders
        train_dataset = SAMDataset(train_X, train_y, norm_target=True, y_min=p_min, y_max=p_max)
        test_dataset = SAMDataset(test_X, test_y, norm_target=True, y_min=p_min, y_max=p_max)

        train_loader = 

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




if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Simple Neural Net on polynomial coefficients")
    parser.add_argument("--infile", "-f", type=str, default="sam_pattern_data.dat.npz",
                        help="Input file name (Default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Size of training batches. There will be (N_data/batch_size) batches in each "\
                             "training epoch.  (Default: %(default)s)")
    parser.add_argument("--n-valid", type=int, default=5,
                        help="Number of partitions for cross-validation (Default: split data into %(default)s groups")


    args = parser.parse_args()
    run(args)

