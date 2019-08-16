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
    splits = fname.split('_')

    n_layer = int(splits[4])
    n_hidden = int(splits[7])
    n_conv_channel = int(splits[10].split('.')[0])

    return (n_layer, n_hidden, n_conv_channel)


# conv channels
trial_channels = np.array([1, 2, 3, 4, 5, 6])
# number hidden nodes
trial_hidden = np.array([2, 4, 6, 8, 12, 16, 18, 20])

# number of CV trials
n_trials = 5
xx, yy = np.meshgrid(trial_channels, trial_hidden, indexing='ij')
# Number of trained parameters (for AIC)
n_params = 7*xx + (9*xx+1)*yy + (yy+1) 
n_sample = 6*884 

all_perf_cv = np.zeros((n_trials, xx.shape[0], xx.shape[1]))
all_perf_tot = np.zeros((xx.shape[0], xx.shape[1]))

all_perf_cv[:] = np.inf
all_perf_tot[:] = np.nan

fnames = sorted(glob.glob("perf_*"))

for fname in fnames:
    n_layer, n_hidden, n_conv_channel = extract_info(fname)

    x_idx = np.digitize(n_conv_channel, trial_channels) - 1
    y_idx = np.digitize(n_hidden, trial_hidden) - 1

    ds = np.load(fname)

    all_perf_cv[:, x_idx, y_idx] = ds['mses_cv']
    all_perf_tot[x_idx, y_idx] = ds['mse_tot'].item()

avg_perf_cv = all_perf_cv.mean(axis=0)

aic = n_sample * np.log(all_perf_tot) + 2 * n_params
aic -= np.nanmin(aic)
fig, ax = plt.subplots()

extent = [0.5, avg_perf_cv.shape[1]+0.5, 0.5, avg_perf_cv.shape[0]+0.5]
norm = plt.Normalize(9,15)

pc = ax.imshow(avg_perf_cv, origin='lower', extent=extent, cmap='plasma_r', norm=plt.Normalize(9,15))

plt.colorbar(pc)

ax.set_xticks(np.arange(avg_perf_cv.shape[1])+1)
ax.set_yticks(np.arange(avg_perf_cv.shape[0])+1)
ax.set_xticklabels(['2', '4', '6', '8', '12', '16', '18', '20'])
ax.set_yticklabels(['1', '2', '3', '4', '5', '6'])

plt.show()


fig, ax = plt.subplots()
pc = ax.imshow(aic, origin='lower', extent=extent, cmap='plasma_r', norm=plt.Normalize(0,10000))

plt.colorbar(pc)

ax.set_xticks(np.arange(avg_perf_cv.shape[1])+1)
ax.set_yticks(np.arange(avg_perf_cv.shape[0])+1)
ax.set_xticklabels(['2', '4', '6', '8', '12', '16', '18', '20'])
ax.set_yticklabels(['1', '2', '3', '4', '5', '6'])

plt.show()

