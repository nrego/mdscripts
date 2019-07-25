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

# Gets fit from normalized coefficients
#   coef is array or torch tensor and shape: (N_data, 5)
def get_fit(norm_coef, p_min, p_range_mat, xvals):

    norm_coef = np.array(norm_coef)
    xvals = np.array(xvals)

    coef = np.dot(norm_coef, p_range_mat) + p_min
    fit = np.dot(coef, xvals).squeeze()

    return xvals[-2], fit


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
    param_str = "Net architecture:\n"
    if args.do_conv:
        param_str += f"Convolutional filters: {args.n_out_channels}\n\n"

    param_str +=f"N hidden layers: {args.n_layers}\n"\
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
    if args.no_run:
        return None, None
    p_min = poly.min(axis=0).astype(np.float32)
    p_max = poly.max(axis=0).astype(np.float32)
    p_range = p_max - p_min
    p_range_mat = torch.tensor(np.diag(p_range))

    # Position of patch indices (only different if we're using an extended grid)
    pos = pos_ext[patch_indices]

    ## Split up input data into training and validation sets
    mses = np.zeros(args.n_valid)
    criterion = nn.MSELoss()
    data_partitions = partition_data(feat_vec, poly, n_groups=args.n_valid)

    ## Train and validate for each round (n_valid rounds)
    for i_round, (train_X, train_y, test_X, test_y) in enumerate(data_partitions):
        
        print("\nCV ROUND {} of {}\n".format(i_round+1, args.n_valid))
        
        if args.do_conv:
            net = SAMConvNet(n_out_channels=args.n_out_channels, n_layers=args.n_layers, 
                             n_hidden=args.n_hidden, n_out=5, drop_out=args.drop_out)
        else:
            net = SAMNet(n_layers=args.n_layers, n_hidden=args.n_hidden, n_out=5, drop_out=args.drop_out)

        train_dataset = DatasetType(train_X, train_y, norm_target=True, y_min=p_min, y_max=p_max)
        test_dataset = DatasetType(test_X, test_y, norm_target=True, y_min=p_min, y_max=p_max)

        ## Round 1 - just on coefficients
        trainer = Trainer(train_dataset, test_dataset, batch_size=args.batch_size,
                          learning_rate=args.learning_rate, epochs=args.n_epochs_first)
        trainer(net, criterion)
        
        del trainer

        print("\nFinished initial\n")
        print("\nBegin training\n")

        ## Round 2 - On full F_v(N) 
        trainer = Trainer(train_dataset, test_dataset, batch_size=args.batch_size, 
                          learning_rate=args.learning_rate, epochs=args.n_epochs_refinement,
                          n_patience=args.n_patience, break_out=args.break_out)

        loss_fn_kwargs = {'p_min': p_min,
                          'p_range_mat': p_range_mat,
                          'xvals': xvals}

        trainer(net, criterion, loss_fn=loss_poly, loss_fn_kwargs=loss_fn_kwargs)
        net.eval()
        pred = net(trainer.test_X).detach()
        net.train()
        
        test_loss = loss_poly(pred, trainer.test_y, criterion, **loss_fn_kwargs)
        print("\n")
        print("Final CV: {:.2f}\n".format(test_loss))
        mses[i_round] = test_loss

    print("\n\nFinal average MSE: {:.2f}".format(mses.mean()))


    ## Final Model: Train on all ##
    print("\n\nFinal Training on Entire Dataset\n")
    print("================================\n")
    
    if args.do_conv:
        net = SAMConvNet(n_out_channels=args.n_out_channels, n_layers=args.n_layers, 
                         n_hidden=args.n_hidden, n_out=5, drop_out=args.drop_out)
    else:
        net = SAMNet(n_layers=args.n_layers, n_hidden=args.n_hidden, n_out=5, drop_out=args.drop_out)

    dataset = DatasetType(feat_vec, poly, norm_target=True, y_min=p_min, y_max=p_max)


    ## Round 1 - just on coefficients
    trainer = Trainer(dataset, dataset, batch_size=args.batch_size,
                      learning_rate=args.learning_rate, epochs=args.n_epochs_first)
    trainer(net, criterion)
    
    del trainer

    print("\nFinished initial\n")
    print("\nBegin training\n")

    ## Round 2 - On full F_v(N) 
    trainer = Trainer(dataset, dataset, batch_size=args.batch_size, 
                      learning_rate=args.learning_rate, epochs=args.n_epochs_refinement,
                      n_patience=args.n_patience, break_out=args.break_out)

    loss_fn_kwargs = {'p_min': p_min,
                      'p_range_mat': p_range_mat,
                      'xvals': xvals}

    trainer(net, criterion, loss_fn=loss_poly, loss_fn_kwargs=loss_fn_kwargs)
    net.eval()
    pred = net(trainer.test_X).detach()
    
    test_loss = loss_poly(pred, trainer.test_y, criterion, **loss_fn_kwargs)
    print("\n")
    print("ALL DATA Final CV: {:.2f}\n".format(test_loss))
    
    embed()

    return trainer, net

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Simple Neural Net on polynomial coefficients")
    parser.add_argument("--n-epochs-first", type=int, default=500,
                       help="Number of epochs for initial training to coefficients (Default: %(default)s)")
    parser.add_argument("--n-epochs-refinement", type=int, default=3000,
                       help="Maximum number of epochs for refinement training to F_v (Default: %(default)s)")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of hidden layers. (Default: %(default)s)")
    parser.add_argument("--n-hidden", type=int, default=18,
                        help="Number of nodes in each hidden layer. (Default: %(default)s)")
    parser.add_argument("--do-conv", action="store_true",
                        help="Do a convolutional neural net (default: false)")
    parser.add_argument("--n-out-channels", type=int, default=4,
                        help="Number of convolutional filters to apply; ignored if not doing CNN (Default: %(default)s)")
    parser.add_argument("--no-run", action="store_true",
                        help="Don't run CV if true")

    args = parser.parse_args()
    feat_vec, energies, poly, beta_phi_stars, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep(args.infile)
    trainer, net = run(args)

