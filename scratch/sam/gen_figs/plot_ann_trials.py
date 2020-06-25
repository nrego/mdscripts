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


## Plot ANN AIC as fn of hyperparams



homedir = os.environ['HOME']

##
ds = np.load('data/sam_ann_ml_trials.npz')
all_nets = ds['all_nets']
all_perf_tot = ds['all_perf_tot']
all_perf_cv = ds['all_perf_cv']
all_n_params = ds['all_n_params']
trial_n_hidden_layer = ds['trial_n_hidden_layer']
trial_n_node_hidden = ds['trial_n_node_hidden']
n_sample = ds['n_sample'].item()

aic = n_sample*np.log(all_perf_tot) + 2*all_n_params
aic -= aic.min()

#min_perf_cv = all_perf_cv.mean(axis=0)

## Tot MSE 
plt.close('all')
fig, ax = plt.subplots()

norm = plt.Normalize(0, 20)
pc = ax.imshow(all_perf_tot.T, origin='lower', cmap='plasma', norm=norm)

ax.set_xticks(np.arange(trial_n_hidden_layer.size))
ax.set_xticklabels(trial_n_hidden_layer)
ax.set_yticks(np.arange(trial_n_node_hidden.size))
ax.set_yticklabels(trial_n_node_hidden)
plt.colorbar(pc)
plt.tight_layout()
plt.savefig('{}/Desktop/ann_mse_tot.png'.format(homedir), transparent=True)


## min CV MSE 
plt.close('all')
fig, ax = plt.subplots()

norm = plt.Normalize(10, 25)
pc = ax.imshow(all_perf_cv.T, origin='lower', cmap='plasma', norm=norm)

ax.set_xticks(np.arange(trial_n_hidden_layer.size))
ax.set_xticklabels(trial_n_hidden_layer)
ax.set_yticks(np.arange(trial_n_node_hidden.size))
ax.set_yticklabels(trial_n_node_hidden)
plt.colorbar(pc)
plt.tight_layout()
plt.savefig('{}/Desktop/ann_mse_cv.png'.format(homedir), transparent=True)


## AIC
plt.close('all')
fig, ax = plt.subplots()

norm = plt.Normalize(0, 30)
pc = ax.imshow(aic.T, origin='lower', cmap='plasma', norm=norm)

ax.set_xticks(np.arange(trial_n_hidden_layer.size))
ax.set_xticklabels(trial_n_hidden_layer)
ax.set_yticks(np.arange(trial_n_node_hidden.size))
ax.set_yticklabels(trial_n_node_hidden)
plt.colorbar(pc)
plt.tight_layout()
plt.savefig('{}/Desktop/ann_aic.png'.format(homedir), transparent=True)


plt.close('all')



