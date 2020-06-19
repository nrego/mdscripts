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

import itertools

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':14})


## Plot ANN AIC as fn of hyperparams



#Get feat vec and augment to get right dimensions
feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('data/sam_pattern_06_06.npz')
n_patch_dim = feat_vec.shape[1]

homedir = os.environ['HOME']

##
ds = np.load('data/sam_cnn_ml_trials.npz')
all_state_dict = ds['all_state_dict']
# n_conv_filters, n_hidden_layer, n_node_hidden
all_hyp_param_array = ds['all_hyp_param_array']


all_perf_tot = ds['all_perf_tot']
all_perf_cv = ds['all_perf_cv']
all_n_params = ds['all_n_params']

trial_n_conv_filters = ds['trial_n_conv_filters']
trial_n_hidden_layer = ds['trial_n_hidden_layer']
trial_n_node_hidden = ds['trial_n_node_hidden']
n_sample = ds['n_sample'].item()

min_perf_cv = all_perf_cv.mean(axis=0)

#aic = n_sample*np.log(min_perf_cv) + 2*all_n_params
aic = n_sample*np.log(min_perf_cv) + 2*all_n_params
aic -= aic.min()



for i_hidden_layer in range(trial_n_hidden_layer.size):

    ## Tot MSE 
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,10))

    norm = plt.Normalize(0, 20)
    pc = ax.imshow(all_perf_tot[:,i_hidden_layer,:], origin='lower', cmap='plasma', norm=norm)

    ax.set_xticks(np.arange(trial_n_node_hidden.size))
    ax.set_xticklabels(trial_n_node_hidden)
    ax.set_yticks(np.arange(trial_n_conv_filters.size))
    ax.set_yticklabels(trial_n_conv_filters)
    plt.colorbar(pc)
    plt.tight_layout()
    plt.savefig('{}/Desktop/cnn_mse_tot_n_hidden_layer_{}.png'.format(homedir, i_hidden_layer+1), transparent=True)


    ## min CV MSE 
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,10))

    norm = plt.Normalize(10, 20)
    pc = ax.imshow(min_perf_cv[:,i_hidden_layer,:], origin='lower', cmap='plasma', norm=norm)

    ax.set_xticks(np.arange(trial_n_node_hidden.size))
    ax.set_xticklabels(trial_n_node_hidden)
    ax.set_yticks(np.arange(trial_n_conv_filters.size))
    ax.set_yticklabels(trial_n_conv_filters)
    plt.colorbar(pc)
    plt.tight_layout()
    plt.savefig('{}/Desktop/cnn_mse_cv_n_hidden_layer_{}.png'.format(homedir, i_hidden_layer+1), transparent=True)



    ## AIC
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,10))

    norm = plt.Normalize(0, 1000)
    pc = ax.imshow(aic[:,i_hidden_layer,:], origin='lower', cmap='plasma', norm=norm)

    ax.set_xticks(np.arange(trial_n_node_hidden.size))
    ax.set_xticklabels(trial_n_node_hidden)
    ax.set_yticks(np.arange(trial_n_conv_filters.size))
    ax.set_yticklabels(trial_n_conv_filters)
    plt.colorbar(pc)
    plt.tight_layout()
    plt.savefig('{}/Desktop/cnn_mse_aic_n_hidden_layer_{}.png'.format(homedir, i_hidden_layer+1), transparent=True)


plt.close('all')

