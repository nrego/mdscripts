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

# Get hyperparams from file name
def extract_info(basename):
    #path = pathlib.Path(fname)
    splits = basename.split('_')

    n_layer = int(splits[4])
    n_hidden = int(splits[7])
    n_conv_channel = int(splits[10].split('.')[0])

    return (n_layer, n_hidden, n_conv_channel)

def extract_n_params(n_hidden_layer, n_hidden, n_channels):

    net = SAMConvNet(n_conv_filters=n_channels, n_hidden_layer=n_hidden_layer, n_hidden=n_hidden, n_out=1)

    n_param = 0
    for param in net.parameters():
        n_param += param.detach().numpy().size


    return n_param


homdir = os.environ['HOME']
tmp_n_layer = 1
headdir = 'eps2'
fnames = sorted(glob.glob("{}/n_layer_{}/n_filter_*/perf_model_*".format(headdir, tmp_n_layer)))

# All combos of hyper params: n_layer, n_hidden_nodes, n_channels
hyp_param_array = np.zeros((len(fnames), 3))
#n_params = np.zeros(len(fnames))

for i, fname in enumerate(fnames):
    basename = os.path.basename(fname)
    this_n_layer, this_n_hidden, this_n_channels = extract_info(basename)
    assert this_n_layer == tmp_n_layer
    hyp_param_array[i] = this_n_layer, this_n_hidden, this_n_channels


trial_hidden = np.unique(hyp_param_array[:,1])
trial_channels = np.unique(hyp_param_array[:,2])

d_hidden = np.diff(trial_hidden)[-1]
d_channel = np.diff(trial_channels)[-1]

trial_hidden = np.append(trial_hidden, trial_hidden[-1]+d_hidden)
trial_channels = np.append(trial_channels, trial_channels[-1]+d_channel)

n_cv = 5
xx, yy = np.meshgrid(trial_channels, trial_hidden, indexing='ij')

n_sample = 6*1794 

all_perf_cv = np.zeros((n_cv, xx.shape[0], xx.shape[1]))
all_perf_tot = np.zeros((xx.shape[0], xx.shape[1]))

# Shape: (N_cv, n_channels, n_hidden_nodes)
all_perf_cv[:] = np.inf
# Shape: (n_channels, n_hidden_nodes)
all_perf_tot[:] = np.nan
all_n_params = np.zeros_like(all_perf_tot)

for i, fname in enumerate(fnames):
    basename = os.path.basename(fname)
    feat = n_layer, n_hidden, n_conv_channel = extract_info(basename)
    n_params = extract_n_params(*feat)

    x_idx = np.digitize(n_conv_channel, trial_channels) - 1
    y_idx = np.digitize(n_hidden, trial_hidden) - 1

    #for i_trial in range(n_trials):
    ds = np.load(fname)
    all_perf_cv[:, x_idx, y_idx] = ds['mses_cv']
    all_perf_tot[x_idx, y_idx] = ds['mse_tot'].item()
    all_n_params[x_idx, y_idx] = n_params

avg_perf_cv = all_perf_cv.mean(axis=0)
var_perf_cv = all_perf_cv.var(axis=0)

norm_var = var_perf_cv / avg_perf_cv
alarm_threshold = 1
high_var = norm_var > alarm_threshold

plt.close('all')
fig, ax = plt.subplots()

norm = plt.Normalize(0, 1)
pc = ax.pcolormesh(xx, yy, norm_var, cmap='plasma_r', norm=norm)

ax.set_xticks(trial_channels)
ax.set_yticks(trial_hidden)

plt.show()

plt.close('all')
fig, ax = plt.subplots(figsize=(10,5))

norm = plt.Normalize(4, 5.5)
pc = ax.pcolormesh(xx, yy, avg_perf_cv, cmap='plasma_r', norm=norm)

plt.colorbar(pc)

ax.set_xticks(trial_channels[:-1] + 0.5*d_channel)
ax.set_yticks(trial_hidden[:-1] + 0.5*d_hidden)
ax.set_xticklabels([r'{}'.format(int(n_channel)) for n_channel in trial_channels[:-1]])
ax.set_yticklabels([r'{}'.format(int(n_hidden)) for n_hidden in trial_hidden[:-1]])
fig.tight_layout()
plt.savefig('{}/Desktop/fig_{}_n_layer_{}'.format(homedir, headdir, tmp_n_layer), transparent=True)

plt.show()

plt.close('all')
aic = n_sample * np.log(all_perf_tot) + 2*all_n_params 
aic -= np.nanmin(aic)

fig, ax = plt.subplots(figsize=(10,5))
pc = ax.pcolormesh(xx, yy, aic, cmap='plasma_r', norm=plt.Normalize(0,20000))

plt.colorbar(pc)

ax.set_xticks(trial_channels[:-1] + 0.5*d_channel)
ax.set_yticks(trial_hidden[:-1] + 0.5*d_hidden)
ax.set_xticklabels([r'{}'.format(int(n_channel)) for n_channel in trial_channels[:-1]])
ax.set_yticklabels([r'{}'.format(int(n_hidden)) for n_hidden in trial_hidden[:-1]])
plt.savefig('{}/Desktop/fig_aic_{}_n_layer_{}'.format(homedir, headdir, tmp_n_layer), transparent=True)

plt.show()

