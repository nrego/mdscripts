import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os, glob

def extract_info(fname):
    with open(fname, 'r') as fin:
        lines = fin.readlines()

    for line in lines:
        splits = line.split(':')

        if splits[0] == 'Convolutional filters':
            n_conv_channel = int(splits[1])

        if splits[0] == 'N hidden layers':
            assert int(splits[1]) == 1

        if splits[0] == 'Nodes per hidden layer':
            n_hidden = int(splits[1])

        if splits[0] == 'Final average MSE':
            mse_cv = float(splits[1])

        if splits[0] == 'ALL DATA Final CV':
            mse_tot = float(splits[1])

    return (n_conv_channel, n_hidden, mse_cv, mse_tot)

def fill_from_dir(pathname, trial_channels, trial_hidden):
    fnames = glob.glob('{}/log_*out*'.format(pathname))

    out_mse_cv = np.zeros((trial_channels.size, trial_hidden.size))
    out_mse_cv[:] = np.inf

    out_mse_tot = np.zeros_like(out_mse_cv)
    out_mse_tot[:] = np.inf

    for fname in fnames:
        n_conv_channel, n_hidden, mse_cv, mse_tot = extract_info(fname)

        bin_channel = np.digitize(n_conv_channel, trial_channels) - 1
        bin_hidden = np.digitize(n_hidden, trial_hidden) - 1

        out_mse_cv[bin_channel, bin_hidden] = mse_cv
        out_mse_tot[bin_channel, bin_hidden] = mse_tot


    return (out_mse_cv, out_mse_tot)


n_trials = 4

trial_channels = np.array([1, 2, 3, 4, 5, 6])
trial_hidden = np.array([4, 8, 12, 16, 20])

xx, yy = np.meshgrid(trial_channels, trial_hidden, indexing='ij')
n_params = 7*xx + (9*xx+1)*yy + (yy+1) 
n_sample = 2*884 - 2

all_perf_cv = np.zeros((n_trials, xx.shape[0], xx.shape[1]))
all_perf_tot = np.zeros_like(all_perf_cv)

for i in range(n_trials):
    pathname = 'trial_{}/logs'.format(i)

    out_mse_cv, out_mse_tot = fill_from_dir(pathname, trial_channels, trial_hidden)

    all_perf_cv[i, ...] = out_mse_cv
    all_perf_tot[i, ...] = out_mse_tot


aic = n_sample * np.log(all_perf_tot[0]) + 2 * n_params
aic -= aic.min()
fig, ax = plt.subplots()
avg_perf = all_perf_cv.mean(axis=0)
extent = [0.5, avg_perf.shape[1]+0.5, 0.5, avg_perf.shape[0]+0.5]
pc = ax.imshow(all_perf_cv[0], origin='lower', extent=extent, cmap='hot_r', norm=plt.Normalize(10,30))

plt.colorbar(pc)

ax.set_xticks(np.arange(avg_perf.shape[1])+1)
ax.set_yticks(np.arange(avg_perf.shape[0])+1)
ax.set_xticklabels(['4', '8', '12', '16', '20'])
ax.set_yticklabels(['1', '2', '3', '4', '5', '6'])

plt.show()

fig, ax = plt.subplots()
pc = ax.imshow(aic, origin='lower', extent=extent, cmap='hot_r')

plt.colorbar(pc)

ax.set_xticks(np.arange(avg_perf.shape[1])+1)
ax.set_yticks(np.arange(avg_perf.shape[0])+1)
ax.set_xticklabels(['4', '8', '12', '16', '20'])
ax.set_yticklabels(['1', '2', '3', '4', '5', '6'])

plt.show()




