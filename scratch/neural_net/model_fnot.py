import numpy as np

from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import argparse

def loss_fnot(net_out, target, criterion, emin, erange):
    pred = (net_out * erange) + emin
    act = (target * erange) + emin

    loss = criterion(pred, act)

    return loss

class FnotModel(NNModel):


    prog='Predict F_v(0) from pattern'
    description = '''\
Construct and train NN to predict f (F_v(0)) from SAM surface patterns


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''

    def __init__(self):
        super(FnotModel, self).__init__()

    def add_args(self, parser):
        tgroup = parser.add_argument_group("Training Options")
        tgroup.add_argument("--n-epochs", type=int, default=3000,
                           help="Number of training epochs (Default: %(default)s)")
        tgroup.add_argument("--break-out", type=float, 
                           help="Stop training if CV MSE goes below this (Default: No break-out)")
    def process_args(self, args):
        # 5 polynomial coefficients for 4th degree polynomial
        self.n_epochs = args.n_epochs
        self.break_out = args.break_out

        feat_vec, energies, poly, beta_phi_stars, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep(args.infile)
        
        if self.augment_data:
            feat_vec, energies = augment_data(feat_vec, energies)

        self.feat_vec = feat_vec
        self.energies = energies
        self.pos_ext = pos_ext
        self.patch_indices = patch_indices

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
                    f"max epochs: {self.n_epochs}\n"\
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

        emin = self.energies.min()
        emax = self.energies.max()
        erange = emax - emin

        # Position of patch indices (only different if we're using an extended grid)
        pos = self.pos_ext[self.patch_indices]

        ## Split up input data into training and validation sets
        mses = np.zeros(self.n_valid)
        criterion = nn.MSELoss()
        data_partitions = partition_data(self.feat_vec, self.energies, n_groups=self.n_valid)

        ## Train and validate for each round (n_valid rounds)
        for i_round, (train_X, train_y, test_X, test_y) in enumerate(data_partitions):
            
            print("\nCV ROUND {} of {}\n".format(i_round+1, self.n_valid))
            print("\nBegin training\n")

            if self.do_conv:
                net = SAMConvNet(n_out_channels=self.n_out_channels, n_layers=self.n_layers, 
                                 n_hidden=self.n_hidden, n_out=1, drop_out=self.drop_out)
            else:
                net = SAMNet(n_layers=self.n_layers, n_hidden=self.n_hidden, n_out=1, drop_out=self.drop_out)

            if torch.cuda.is_available():
                net = net.cuda()

            train_dataset = DatasetType(train_X, train_y, norm_target=True, y_min=emin, y_max=emax)
            test_dataset = DatasetType(test_X, test_y, norm_target=True, y_min=emin, y_max=emax)

            trainer = Trainer(train_dataset, test_dataset, batch_size=self.batch_size,
                              learning_rate=self.learning_rate, n_patience=self.n_patience, 
                              epochs=self.n_epochs)
            
            loss_fn_kwargs = {'emin': emin,
                              'erange': erange}

            trainer(net, criterion, loss_fn=loss_fnot, loss_fn_kwargs=loss_fn_kwargs)
            
            net.eval()
            pred = net(trainer.test_X).detach()
            self.cv_nets.append(net)
            net.train()
            
            test_loss = loss_fnot(pred, trainer.test_y, criterion, **loss_fn_kwargs)
            print("\n")
            print("Final CV: {:.2f}\n".format(test_loss))
            mses[i_round] = test_loss

        print("\n\nFinal average MSE: {:.2f}".format(mses.mean()))


        ## Final Model: Train on all ##
        print("\n\nFinal Training on Entire Dataset\n")
        print("================================\n")

        print("\nBegin training\n")
        
        if self.do_conv:
            net = SAMConvNet(n_out_channels=self.n_out_channels, n_layers=self.n_layers, 
                             n_hidden=self.n_hidden, n_out=1, drop_out=self.drop_out)
        else:
            net = SAMNet(n_layers=self.n_layers, n_hidden=self.n_hidden, n_out=1, drop_out=self.drop_out)

        dataset = DatasetType(self.feat_vec, self.energies, norm_target=True, y_min=emin, y_max=emax)


        trainer = Trainer(dataset, dataset, batch_size=self.batch_size,
                          learning_rate=self.learning_rate, epochs=self.n_epochs)
        
        trainer(net, criterion, loss_fn=loss_fnot, loss_fn_kwargs=loss_fn_kwargs)

        net.eval()
        pred = net(trainer.test_X).detach()
        
        test_loss = loss_fnot(pred, trainer.test_y, criterion, **loss_fn_kwargs)
        print("\n")
        print("ALL DATA Final CV: {:.2f}\n".format(test_loss))
        
        torch.save(net.state_dict(), 'model_n_layer_{}_n_hidden_{:02d}_n_channel_{:02d}_all.pkl'.format(self.n_layers, self.n_hidden, self.n_out_channels))
        for i, cv_net in enumerate(self.cv_nets):
            torch.save(cv_net.state_dict(), 'model_n_layer_{}_n_hidden_{:02d}_n_channel_{:02d}_rd_{}.pkl'.format(self.n_layers, self.n_hidden, self.n_out_channels, i))

if __name__ == "__main__":

    FnotModel().main()
