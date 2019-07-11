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

    print(
          '''
BEGIN TRAINING
==============
Parameters:

          ''')
    param_str = "Net architecture:\n"\
                f"N hidden layers: {args.n_layers}\n"\
                f"Nodes per hidden layer: {args.n_hidden}\n"\
                "\n"\
                f"learning rate: {args.learning_rate:1.1e}\n"\
                f"max epochs: {args.n_epochs_refinement}\n"\
                "\n"\
                f"Number of CV groups: {args.n_valid}\n"\
                f"patience: {args.n_patience} epochs"\
                "\n"\
                "\n"

    print(param_str)
    
    # Load input features
    feat_vec, energies, poly, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep(args.infile)

    p_min = poly.min(axis=0).astype(np.float32)
    p_max = poly.max(axis=0).astype(np.float32)
    p_range = p_max - p_min
    p_range_mat = torch.tensor(np.diag(p_range))

    # Position of patch indices (only different if we're using an extended grid)
    pos = pos_ext[patch_indices]


    net = SAMNet(n_layers=args.n_layers, n_hidden=args.n_hidden, n_out=5)

    train_dataset = SAMDataset(feat_vec, poly, norm_target=True, y_min=p_min, y_max=p_max)
    test_dataset = SAMDataset(feat_vec, poly, norm_target=True, y_min=p_min, y_max=p_max)
    
    ## Round 1 - train on normalized coefficients
    trainer = Trainer(train_dataset, test_dataset, batch_size=args.batch_size, 
                      learning_rate=args.learning_rate, epochs=args.n_epochs_first)
    
    criterion = torch.nn.MSELoss()
    trainer(net, criterion)
    del trainer

    print('''\nDone with initial training
               ==========================\n''')

    ## Round 2 - refinement
    data_partitions = partition_data(feat_vec, poly, n_groups=args.n_valid)
    mses = np.zeros(args.n_valid)
    for i_round, (train_X, train_y, test_X, test_y) in enumerate(data_partitions):
        print("\nCV ROUND {} of {}\n".format(i_round+1, args.n_valid))

        train_dataset = SAMDataset(train_X, train_y, norm_target=True, y_min=p_min, y_max=p_max)
        test_dataset = SAMDataset(test_X, test_y, norm_target=True, y_min=p_min, y_max=p_max)

        trainer = Trainer(train_dataset, test_dataset, batch_size=args.batch_size, 
                          learning_rate=args.learning_rate, epochs=args.n_epochs_refinement,
                          n_patience=args.n_patience)

        loss_fn_kwargs = {'p_min': p_min,
                          'p_range_mat': p_range_mat,
                          'xvals': xvals}

        trainer(net, criterion, loss_fn=loss_poly, loss_fn_kwargs=loss_fn_kwargs)

        pred = net(trainer.test_X).detach()
        test_loss = loss_poly(pred, trainer.test_y, criterion, **loss_fn_kwargs)
        print("\n")
        print("Final CV: {:.2f}\n".format(test_loss))
        mses[i_round] = test_loss

    print("\n\nFinal average MSE: {:.2f}".format(mses.mean()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Simple Neural Net on polynomial coefficients")
    parser.add_argument("--infile", "-f", type=str, default="sam_pattern_data.dat.npz",
                        help="Input file name (Default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Size of training batches. There will be (N_data/batch_size) batches in each "\
                             "training epoch.  (Default: %(default)s)")
    parser.add_argument("--n-valid", type=int, default=5,
                        help="Number of partitions for cross-validation (Default: split data into %(default)s groups")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for training (Default: %(default)s)")
    parser.add_argument("--n-epochs-first", type=int, default=1000,
                        help="Number of epochs for initial training to coefficients (Default: %(default)s)")
    parser.add_argument("--n-epochs-refinement", type=int, default=3000,
                        help="Maximum number of epochs for refinement training to F_v (Default: %(default)s)")
    parser.add_argument("--n-patience", type=int, default=None,
                        help="Maximum number of epochs to tolerate where CV performance decreases before breaking out of training."\
                             "Default: No break-out.")
    parser.add_argument('--n-layers', type=int, default=3,
                        help="Number of hidden layers. (Default: %(default)s)")
    parser.add_argument('--n-hidden', type=int, default=18,
                        help="Number of nodes in each hidden layer. (Default: %(default)s)")

    args = parser.parse_args()
    run(args)

