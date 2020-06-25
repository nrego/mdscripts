# Analyze results (in ml_tests) for model_fnot (or epsilon training)

import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os, glob, pathlib

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':14})


## Extract NN training data (ANN, CNN, whatever)
##  Run from within ml_tests directory


# Get hyperparams from file name
def extract_info(basename):
    
    splits = basename.split('_')

    n_hidden_layer = int(splits[5])
    n_node_hidden = int(splits[9])


    return (n_hidden_layer, n_node_hidden)

# Constructs net (with given hyper params) and returns it and its number of params
def extract_n_params(n_hidden_layer, n_node_hidden, n_patch_dim):

    net = SAMNet(n_patch_dim=n_patch_dim, n_hidden_layer=n_hidden_layer, n_node_hidden=n_node_hidden, n_out=1)

    n_param = 0
    for param in net.parameters():
        n_param += param.detach().numpy().size


    return n_param, net

def find_best_trial(path, base_mse, thresh=0.01, choices=['aug_ann1', 'aug_ann2', 'aug_ann3']):

    min_mse_tot = np.inf
    min_mses_cv = None
    best_headdir = None

    mses_cv = np.array([])

    for headdir in choices:

        this_path = pathlib.Path(headdir, *path.parts)
        ds = np.load(this_path)
        this_mses_cv = ds['mses_cv']
        this_mse_tot = ds['mse_tot'].item()

        # Mask CV'd values that are too large
        mask_cv = (this_mses_cv / base_mse) < thresh

        mses_cv = np.append(mses_cv, this_mses_cv[mask_cv])

        #print("headdir: {}  mse: {:.2f}".format(headdir, this_mse_tot))
        
        if this_mse_tot < min_mse_tot:

            min_mse_tot = this_mse_tot
            min_mses_cv = this_mses_cv
            best_headdir = headdir


    return best_headdir, min_mse_tot, mses_cv


### EXTRACT PATTERN DATA ###
############################ 

#Get feat vec and augment to get right dimensions
feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('sam_pattern_06_06.npz', embed_pos_ext=True)
n_patch_dim = feat_vec.shape[1]
n_sample = feat_vec.shape[0]

feat_vec, energies = hex_augment_data(feat_vec, energies, pos_ext, patch_indices)
dataset = SAMDataset(feat_vec, energies)

### EXTRACT ANN HYPERPARAMS AND PERFS ###
#########################################

fnames = sorted(glob.glob("aug_ann1/n_layer_*/perf_model_*"))

hyp_param_array = np.zeros((len(fnames), 2), dtype=int)

for i, fname in enumerate(fnames):
    basename = os.path.basename(fname)
    this_n_hidden_layer, this_n_node_hidden = extract_info(basename)
    hyp_param_array[i] = this_n_hidden_layer, this_n_node_hidden

trial_n_hidden_layer = np.unique(hyp_param_array[:,0])
trial_n_node_hidden = np.unique(hyp_param_array[:,1])

xx, yy = np.meshgrid(trial_n_hidden_layer, trial_n_node_hidden, indexing='ij')


all_perf_cv = np.zeros((xx.shape[0], xx.shape[1]))
se_all_perf_cv = np.zeros_like(all_perf_cv)
all_perf_tot = np.zeros_like(all_perf_cv)

# Shape: (N_cv, n_channels, n_hidden_nodes)
all_perf_cv[:] = np.inf
# Shape: (n_channels, n_hidden_nodes)
all_perf_tot[:] = np.nan
all_perf_test = np.zeros_like(all_perf_tot)
all_perf_test[:] = np.nan

all_n_params = np.zeros_like(all_perf_tot)
all_nets = np.zeros((xx.shape[0], xx.shape[1]), dtype=object)

###################################################################


###### FIND MINIMUM PERFORMANCE MODELS #########
################################################

for i, fname in enumerate(fnames):

    path = pathlib.Path(*pathlib.Path(fname).parts[1:])

    feat = n_hidden_layer, n_node_hidden = hyp_param_array[i]
    n_params, net = extract_n_params(*feat, n_patch_dim)

    # Get a baseline performance
    pred = net(dataset.X).detach().numpy().squeeze()
    err = energies - pred
    base_mse = np.mean(err**2)

    x_idx = np.digitize(n_hidden_layer, trial_n_hidden_layer) - 1
    y_idx = np.digitize(n_node_hidden, trial_n_node_hidden) - 1

    #for i_trial in range(n_trials):
    best_trial_dir, mse_tot, mses_cv = find_best_trial(path, base_mse)
    
    all_perf_cv[x_idx, y_idx] = mses_cv.mean()
    se_all_perf_cv[x_idx, y_idx] = mses_cv.std(ddof=1)
    all_perf_tot[x_idx, y_idx] = mse_tot
    all_n_params[x_idx, y_idx] = n_params

    # Extract and load net
    state_fname = 'model_n_hidden_layer_{}_n_node_hidden_{:02d}_n_conv_filters_04_all.pkl'.format(n_hidden_layer, n_node_hidden)
    state_path = pathlib.Path(best_trial_dir, path.parts[0], state_fname)
    net.load_state_dict(torch.load(state_path, map_location='cpu'))

    pred = net(dataset.X).detach().numpy().squeeze()
    mse = np.mean((energies - pred)**2)
    aic = n_sample*np.log(mse) + 2*n_params

    print("n_hidden_layer: {}. n_node_hidden: {} mse: {:.2f}  mse (CV'd): {:.2f} ({:.2}) n_params: {}".format(n_hidden_layer, n_node_hidden, mse, mses_cv.mean(), mses_cv.std(ddof=1), n_params))

    all_perf_test[x_idx, y_idx] = mse

    all_nets[x_idx, y_idx] = net


avg_perf_cv = all_perf_cv.mean(axis=0)
min_perf_cv = all_perf_cv.min(axis=0)

np.savez_compressed("sam_ann_ml_trials", all_nets=all_nets, all_perf_tot=all_perf_test, 
                    all_perf_cv=all_perf_cv, trial_n_hidden_layer=trial_n_hidden_layer, 
                    trial_n_node_hidden=trial_n_node_hidden, n_sample=n_sample, all_n_params=all_n_params)


