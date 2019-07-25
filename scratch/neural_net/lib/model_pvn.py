import numpy as np

from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import argparse


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


class PvNModel(NNModel):

    xvals = np.arange(0,124)
    xvals = np.array([xvals**4,
                      xvals**3,
                      xvals**2,
                      xvals**1,
                      xvals**0])
    xvals = torch.tensor(xvals.astype(np.float32))

    prog='Predict PvN from pattern'
    description = '''\
Construct and train NN to predict PvN (F_v(N)) from SAM surface patterns


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''

    def __init__(self):
        self.super(PvNModel, self).__init__()

        self.feat_vec = None
        self.coefs = None

        self.n_epoch_init = None
        self.n_epoch_refinement = None


    def add_args(self, parser):
        tgroup = parser.add_argument_group("Training Options")
        tgroup.add_argument("--n-epochs-first", type=int, default=500,
                           help="Number of epochs for initial training to coefficients (Default: %(default)s)")
        tgroup.add_argument("--n-epochs-refinement", type=int, default=3000,
                           help="Maximum number of epochs for refinement training to F_v (Default: %(default)s)")

    def process_args(self, args):
        # 5 polynomial coefficients for 4th degree polynomial
        self.n_hidden = 5
        self.n_epoch_init = args.n_epochs_first
        self.n_epoch_refinement = args.n_epochs_refinement

        feat_vec, energies, poly, beta_phi_stars, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep(args.infile)
        
        self.feat_vec = feat_vec
        self.coefs = args.poly

    def run(self):
        print(
              '''
    BEGIN TRAINING
    ==============
    Parameters:

              ''')
        param_str = "Net architecture:\n"
        if self.do_conv:
            param_str += f"Convolutional filters: {self.n_out_channels}\n\n"

        param_str +=f"N hidden layers: {self.n_layers}\n"\
                    f"Nodes per hidden layer: {self.n_hidden}\n"\
                    "\n"\
                    f"learning rate: {self.learning_rate:1.1e}\n"\
                    f"max epochs: {self.n_epochs_refinement}\n"\
                    "\n"\
                    f"Number of CV groups: {self.n_valid}\n"\
                    f"patience: {self.n_patience} epochs"\
                    "\n"\
                    "\n"

        print(param_str)

        DatasetType = SAMConvDataset if self.do_conv else SAMDataset
        NetType = SAMConvNet if self.do_conv else SAMNet

        
        if self.no_run:
            return

        p_min = self.poly.min(axis=0).astype(np.float32)
        p_max = self.poly.max(axis=0).astype(np.float32)
        p_range = p_max - p_min
        p_range_mat = torch.tensor(np.diag(p_range))

        # Position of patch indices (only different if we're using an extended grid)
        pos = pos_ext[patch_indices]

        ## Split up input data into training and validation sets
        mses = np.zeros(self.n_valid)
        criterion = nn.MSELoss()
        data_partitions = partition_data(self.feat_vec, self.poly, n_groups=self.n_valid)

        ## Train and validate for each round (n_valid rounds)
        for i_round, (train_X, train_y, test_X, test_y) in enumerate(data_partitions):
            
            print("\nCV ROUND {} of {}\n".format(i_round+1, self.n_valid))
            
            if self.do_conv:
                net = SAMConvNet(n_out_channels=self.n_out_channels, n_layers=self.n_layers, 
                                 n_hidden=self.n_hidden, n_out=self.n_out_channels, drop_out=self.drop_out)
            else:
                net = SAMNet(n_layers=self.n_layers, n_hidden=self.n_hidden, n_out=self.n_out_channels, drop_out=self.drop_out)

            train_dataset = DatasetType(train_X, train_y, norm_target=True, y_min=p_min, y_max=p_max)
            test_dataset = DatasetType(test_X, test_y, norm_target=True, y_min=p_min, y_max=p_max)

            ## Round 1 - just on coefficients
            trainer = Trainer(train_dataset, test_dataset, batch_size=self.batch_size,
                              learning_rate=self.learning_rate, epochs=self.n_epochs_first)
            trainer(net, criterion)
            
            del trainer

            print("\nFinished initial\n")
            print("\nBegin training\n")

            ## Round 2 - On full F_v(N) 
            trainer = Trainer(train_dataset, test_dataset, batch_size=self.batch_size, 
                              learning_rate=self.learning_rate, epochs=self.n_epochs_refinement,
                              n_patience=self.n_patience, break_out=self.break_out)

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
        
        if self.do_conv:
            net = SAMConvNet(n_out_channels=self.n_out_channels, n_layers=self.n_layers, 
                             n_hidden=self.n_hidden, n_out=5, drop_out=self.drop_out)
        else:
            net = SAMNet(n_layers=self.n_layers, n_hidden=self.n_hidden, n_out=5, drop_out=self.drop_out)

        dataset = DatasetType(feat_vec, poly, norm_target=True, y_min=p_min, y_max=p_max)


        ## Round 1 - just on coefficients
        trainer = Trainer(dataset, dataset, batch_size=self.batch_size,
                          learning_rate=self.learning_rate, epochs=self.n_epochs_first)
        trainer(net, criterion)
        
        del trainer

        print("\nFinished initial\n")
        print("\nBegin training\n")

        ## Round 2 - On full F_v(N) 
        trainer = Trainer(dataset, dataset, batch_size=self.batch_size, 
                          learning_rate=self.learning_rate, epochs=self.n_epochs_refinement,
                          n_patience=self.n_patience, break_out=self.break_out)

        loss_fn_kwargs = {'p_min': p_min,
                          'p_range_mat': p_range_mat,
                          'xvals': xvals}

        trainer(net, criterion, loss_fn=loss_poly, loss_fn_kwargs=loss_fn_kwargs)
        net.eval()
        pred = net(trainer.test_X).detach()
        
        test_loss = loss_poly(pred, trainer.test_y, criterion, **loss_fn_kwargs)
        print("\n")
        print("ALL DATA Final CV: {:.2f}\n".format(test_loss))
        
        

        




