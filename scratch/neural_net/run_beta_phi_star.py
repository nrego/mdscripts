## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import argparse


def run(args):

    print(
          '''
BEGIN TRAINING
==============
Parameters:

          ''')
    param_str = "Net architecture:\n"
    if args.do_conv:
        param_str += f"Convolutional filters: {args.n_conv_filters}\n\n"

    param_str +=f"N hidden layers: {args.n_hidden_layer}\n"\
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

    DatasetType = SAMConvDataset if args.do_conv else SAMDataset
    NetType = SAMConvNet if args.do_conv else SAMNet

    # Load input features
    feat_vec, energies, poly, beta_phi_stars, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep(args.infile)
    #shuffle = np.random.permutation(beta_phi_stars.shape[0])
    #beta_phi_stars = beta_phi_stars[shuffle]
    # Position of patch indices (only different if we're using an extended grid)
    pos = pos_ext[patch_indices]

    ## Split up input data into training and validation sets
    mses = np.zeros(args.n_valid)
    criterion = nn.MSELoss()
    data_partitions = partition_data(feat_vec, beta_phi_stars, n_groups=args.n_valid)

    ## Train and validate for each round (n_valid rounds)
    for i_round, (train_X, train_y, test_X, test_y) in enumerate(data_partitions):
        
        print("\nCV ROUND {} of {}\n".format(i_round+1, args.n_valid))
        
        if args.do_conv:
            net = SAMConvNet(n_conv_filters=args.n_conv_filters, n_hidden_layer=args.n_hidden_layer, 
                             n_hidden=args.n_hidden, n_out=36, drop_out=args.drop_out)
        else:
            net = SAMNet(n_hidden_layer=args.n_hidden_layer, n_hidden=args.n_hidden, n_out=36, drop_out=args.drop_out)

        train_dataset = DatasetType(train_X, train_y)
        test_dataset = DatasetType(test_X, test_y)

        trainer = Trainer(train_dataset, test_dataset, batch_size=args.batch_size, 
                          learning_rate=args.learning_rate, epochs=args.n_epochs_refinement,
                          n_patience=args.n_patience, break_out=args.break_out)


        trainer(net, criterion)
        net.eval()
        pred = net(trainer.test_X).detach()
        net.train()
        
        test_loss = criterion(pred, trainer.test_y)
        print("\n")
        print("Final CV: {:.2f}\n".format(test_loss))
        mses[i_round] = test_loss

    print("\n\nFinal average MSE: {:.2f}".format(mses.mean()))


    ## Final Model: Train on all ##
    print("\n\nFinal Training on Entire Dataset\n")
    print("================================\n")
    
    if args.do_conv:
        net = SAMConvNet(n_conv_filters=args.n_conv_filters, n_hidden_layer=args.n_hidden_layer, 
                         n_hidden=args.n_hidden, n_out=36, drop_out=args.drop_out)
    else:
        net = SAMNet(n_hidden_layer=args.n_hidden_layer, n_hidden=args.n_hidden, n_out=36, drop_out=args.drop_out)

    dataset = DatasetType(feat_vec, beta_phi_stars)


 
    trainer = Trainer(dataset, dataset, batch_size=args.batch_size, 
                      learning_rate=args.learning_rate, epochs=args.n_epochs_refinement,
                      n_patience=args.n_patience, break_out=args.break_out)


    trainer(net, criterion)
    net.eval()
    pred = net(trainer.test_X).detach()
    
    test_loss = criterion(pred, trainer.test_y)
    print("\n")
    print("ALL DATA Final CV: {:.2f}\n".format(test_loss))

    return trainer, net

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Neural Net on beta phi stars for each subvolume")
    parser.add_argument("--infile", "-f", type=str, default="sam_pattern_data.dat.npz",
                        help="Input file name (Default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Size of training batches. There will be (N_data/batch_size) batches in each "\
                             "training epoch.  (Default: %(default)s)")
    parser.add_argument("--n-valid", type=int, default=5,
                        help="Number of partitions for cross-validation (Default: split data into %(default)s groups")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for training (Default: %(default)s)")
    parser.add_argument("--drop-out", type=float, default=0.0,
                        help="Drop-out rate for each layer (Default: %(default)s)")
    parser.add_argument("--n-epochs-first", type=int, default=500,
                        help="Number of epochs for initial training to coefficients (Default: %(default)s)")
    parser.add_argument("--n-epochs-refinement", type=int, default=3000,
                        help="Maximum number of epochs for refinement training to F_v (Default: %(default)s)")
    parser.add_argument("--n-patience", type=int, default=None,
                        help="Maximum number of epochs to tolerate where CV performance decreases before breaking out of training."\
                             "Default: No break-out.")
    parser.add_argument("--break-out", type=float, default=None,
                        help="Break out of training if CV MSE falls below this value."\
                             "Default: No break-out.")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of hidden layers. (Default: %(default)s)")
    parser.add_argument("--n-hidden", type=int, default=18,
                        help="Number of nodes in each hidden layer. (Default: %(default)s)")
    parser.add_argument("--do-conv", action="store_true",
                        help="Do a convolutional neural net (default: false)")
    parser.add_argument("--n-out-channels", type=int, default=4,
                        help="Number of convolutional filters to apply; ignored if not doing CNN (Default: %(default)s)")

    args = parser.parse_args()
    trainer, net = run(args)

