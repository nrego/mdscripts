import numpy as np

from scratch.neural_net.lib import *
from scratch.sam.util import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import argparse

from sklearn import linear_model
import sys

def loss_fnot_norm(net_out, target, criterion, emin, erange):
    pred = (net_out * erange) + emin
    act = (target * erange) + emin

    loss = criterion(pred, act)

    return loss

def loss_fnot(net_out, target, criterion):
    loss = criterion(net_out, target)

    return loss

def get_err(X, y, weights=None, fit_intercept=False):
    
    if weights is None:
        weights = np.ones_like(y)
    
    reg = linear_model.LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X, y, sample_weight=weights)

    pred = reg.predict(X)
    err = y - pred

    return err


class FnotDriver(NNDriver):


    prog='Predict delta_f from pattern'
    description = '''\
Construct and train NN to predict f (F_v(0)) from SAM surface patterns


-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''

    def __init__(self):
        super(FnotDriver, self).__init__()

    def add_args(self, parser):
        tgroup = parser.add_argument_group("f-model specific options (including epsilon or delta training on OLS residuals)")
        tgroup.add_argument("--eps-m1", action="store_true",
                           help="If true, perform epsilon training on errors from linear regression on k_ch3, rather than actual energies (default: False)")
        tgroup.add_argument("--eps-m2", action="store_true",
                           help="If true, perform epsilon training on errors from linear regression on (k_ch3, n_mm) rather than actual energies (default: False)")
        tgroup.add_argument("--skip-cv", action="store_true", 
                           help="If true, skip the N-fold CV and just fit on entire dataset (default: False, perform CV)")

    def process_args(self, args):

        ## Extract our sam datasets (plus a bunch of extra info that might or might not be used)
        feat_vec, patch_indices, pos_ext, energies, delta_e, dg_bind, weights, ols_feat, states = load_and_prep(args.infile)
        
        
        y = dg_bind

        # Only used for epsilon training 
        feat_idx = np.array([2,3,4])
        if args.eps_m1:
            err = get_err(ols_feat[:,feat_idx[0]].reshape(-1,1), delta_e, weights)
            mse = np.mean(err**2)
            print("Doing epsilon on M1 with MSE: {:.2f}".format(mse))
            y = err

        if args.eps_m2:
            err = get_err(ols_feat[:,feat_idx], delta_e, weights)
            mse = np.mean(err**2)
            print("Doing epsilon on M2 with MSE: {:.2f}".format(mse))
            y = err


        if self.augment_data:
            feat_vec, y = hex_augment_data(feat_vec, y, pos_ext, patch_indices)

        self.y = y
        self.feat_vec = feat_vec
        
        self.pos_ext = pos_ext
        self.patch_indices = patch_indices

        self.n_patch_dim = feat_vec.shape[1]

    def run(self):
        print(
              '''
    BEGIN TRAINING
    ==============
    Parameters:

              ''')
        param_str = "Net architecture:\n"
        if self.do_conv:
            param_str += f"Convolutional filters: {self.n_conv_filters}\n\n"

        param_str +=f"N hidden layers: {self.n_hidden_layer}\n"\
                    f"Nodes per hidden layer: {self.n_node_hidden}\n"\
                    f"Nodes per feature layer: {self.n_node_feature}\n"\
                    "\n"\
                    f"learning rate: {self.learning_rate:1.1e}\n"\
                    f"max epochs: {self.n_epochs}\n"\
                    "\n"\
                    f"Number of CV groups: {self.n_valid}\n"\
                    f"patience: {self.n_patience} epochs"\
                    "\n"\
                    "\n"

        print(param_str)
        sys.stdout.flush()

        # The datasets simply add empty dimensions to make compatible with CNNs
        DatasetType = SAMConvDataset if self.do_conv else SAMDataset


        if self.do_conv:
            NetType = SAMConvNet #if self.n_node_hidden_layer > 0 else SAMFixedConvNet
        else:
            NetType = SAMNet

        if self.no_run:
            return

        emin = self.y.min()
        emax = self.y.max()
        erange = emax - emin

        ## Split up input data into training and validation sets, and set up our loss function
        mses = np.zeros(self.n_valid)
        criterion = nn.MSELoss()
        data_partitions = partition_data(self.feat_vec, self.y, n_groups=self.n_valid)

        #loss_fn_kwargs = {'emin': emin,
        #                  'erange': erange}

        loss_fn_kwargs = {}
                          
        ## BEGIN TRAINING ##
        ####################

        ## DO CROSS VALIDATION ##
        for i_round, (train_X, train_y, test_X, test_y) in enumerate(data_partitions):
            if self.skip_cv:
                print("\nSkipping CV...")
                break

            print("\nCV ROUND {} of {}\n".format(i_round+1, self.n_valid))
            print("\nBegin training\n")
            sys.stdout.flush()
            
            if self.do_conv:
                net = NetType(n_conv_filters=self.n_conv_filters, n_hidden_layer=self.n_hidden_layer, 
                                 n_node_hidden=self.n_node_hidden, n_node_feature=self.n_node_feature, n_out=1, drop_out=self.drop_out)
            else:
                net = SAMNet(n_patch_dim=self.n_patch_dim, n_hidden_layer=self.n_hidden_layer, n_node_hidden=self.n_node_hidden, 
                             n_node_feature=self.n_node_feature, n_out=1, drop_out=self.drop_out)

            if torch.cuda.is_available():
                print("\n(GPU detected)")
                net = net.cuda()
            else:
                print("\n(No GPU detected)")

            train_dataset = DatasetType(train_X, train_y, norm_target=False, y_min=emin, y_max=emax)
            test_dataset = DatasetType(test_X, test_y, norm_target=False, y_min=emin, y_max=emax)

            trainer = Trainer(train_dataset, test_dataset, batch_size=self.batch_size,
                              learning_rate=self.learning_rate, n_patience=self.n_patience, 
                              epochs=self.n_epochs)

            trainer(net, criterion, loss_fn=loss_fnot, loss_fn_kwargs=loss_fn_kwargs)
            
            net.eval()
            pred = net(trainer.test_X).detach()
            #self.cv_nets.append(net)
            net.train()
            
            test_loss = loss_fnot(pred, trainer.test_y, criterion, **loss_fn_kwargs)
            print("\n")
            print("Final CV: {:.2f}\n".format(test_loss))
            mses[i_round] = test_loss

            # Save this net
            torch.save(net.state_dict(), 'model_n_layer_{}_n_node_hidden_{:02d}_n_channel_{:02d}_rd_{}.pkl'.format(self.n_hidden_layer, self.n_node_hidden, self.n_conv_filters, i_round))

            del net, trainer, train_dataset, test_dataset, train_X, train_y, test_X, test_y

        print("\n\nFinal average MSE: {:.2f}".format(mses.mean()))

        #np.savez_compressed('perf_model_n_layer_{}_n_node_hidden_{:02d}_n_channel_{:02d}'.format(self.n_hidden_layer, self.n_node_hidden, self.n_conv_filters),
        #        mses_cv=mses)
        ## Final Model: Train on all ##
        print("\n\nFinal Training on Entire Dataset\n")
        print("================================\n")

        print("\nBegin training\n")

        if self.do_conv:
            net = NetType(n_conv_filters=self.n_conv_filters, n_hidden_layer=self.n_hidden_layer, 
                             n_node_hidden=self.n_node_hidden, n_node_feature=self.n_node_feature, n_out=1, drop_out=self.drop_out)
        else:
            net = SAMNet(n_patch_dim=self.n_patch_dim, n_hidden_layer=self.n_hidden_layer, n_node_hidden=self.n_node_hidden, 
                         n_node_feature=self.n_node_feature, n_out=1, drop_out=self.drop_out)

        if torch.cuda.is_available():
            print("\n(GPU detected)")
            net = net.cuda()
        else:
            print("\n(No GPU detected)")

        sys.stdout.flush()

        dataset = DatasetType(self.feat_vec, self.y, norm_target=False, y_min=emin, y_max=emax)


        trainer = Trainer(dataset, dataset, batch_size=self.batch_size,
                          learning_rate=self.learning_rate, epochs=int(self.n_epochs*0.5), n_patience=self.n_patience)
        
        trainer(net, criterion, loss_fn=loss_fnot, loss_fn_kwargs=loss_fn_kwargs)

        net.eval()
        pred = net(trainer.test_X).detach()
        
        test_loss = loss_fnot(pred, trainer.test_y, criterion, **loss_fn_kwargs)
        print("\n")
        print("ALL DATA Final CV: {:.2f}\n".format(test_loss))

        if torch.cuda.is_available():
            test_loss = test_loss.cpu()
            
        np.savez_compressed('perf_model_n_hidden_layer_{}_n_node_hidden_{:02d}_n_conv_filters_{:02d}'.format(self.n_hidden_layer, self.n_node_hidden, self.n_conv_filters),
                mses_cv=mses, mse_tot=test_loss)

        torch.save(net.state_dict(), 'model_n_hidden_layer_{}_n_node_hidden_{:02d}_n_conv_filters_{:02d}_all.pkl'.format(self.n_hidden_layer, self.n_node_hidden, self.n_conv_filters))

if __name__ == "__main__":

    FnotDriver().main()
