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


def extract_n_params(n_hidden_layer, n_node_hidden, n_patch_dim):

    net = SAMNet(n_patch_dim=n_patch_dim, n_hidden_layer=n_hidden_layer, n_node_hidden=n_node_hidden, n_out=1)

    n_param = 0
    for param in net.parameters():
        n_param += param.detach().numpy().size


    return n_param, net

#Get feat vec and augment to get right dimensions
feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('sam_pattern_06_06.npz')
n_patch_dim = feat_vec.shape[1]

homdir = os.environ['HOME']

##

fnames = sorted(glob.glob("n_layer_*/perf_model_*"))


hyp_param_array = np.zeros((len(fnames), 2), dtype=int)

for i, fname in enumerate(fnames):
    basename = os.path.basename(fname)
    this_n_hidden_layer, this_n_node_hidden = extract_info(basename)
    hyp_param_array[i] = this_n_hidden_layer, this_n_node_hidden


trial_n_hidden_layer = np.unique(hyp_param_array[:,0])
trial_n_node_hidden = np.unique(hyp_param_array[:,1])

d_hidden = np.diff(trial_n_hidden_layer)[-1]
d_node = np.diff(trial_n_node_hidden)[-1]

trial_n_hidden_layer = np.append(trial_n_hidden_layer, trial_n_hidden_layer[-1]+d_hidden)
trial_n_node_hidden = np.append(trial_n_node_hidden, trial_n_node_hidden[-1]+d_node)

n_cv = 5
xx, yy = np.meshgrid(trial_n_hidden_layer, trial_n_node_hidden, indexing='ij')

#n_sample = 6*1794 

all_perf_cv = np.zeros((n_cv, xx.shape[0], xx.shape[1]))
all_perf_tot = np.zeros((xx.shape[0], xx.shape[1]))

# Shape: (N_cv, n_channels, n_hidden_nodes)
all_perf_cv[:] = np.inf
# Shape: (n_channels, n_hidden_nodes)
all_perf_tot[:] = np.nan
all_perf_test = np.zeros_like(all_perf_tot)
all_perf_test[:] = np.nan

all_n_params = np.zeros_like(all_perf_tot)
all_nets = np.zeros((xx.shape[0], xx.shape[1]), dtype=object)
for i, fname in enumerate(fnames):
    basename = os.path.basename(fname)
    dirname = os.path.dirname(fname)

    feat = n_hidden_layer, n_node_hidden = hyp_param_array[i]
    n_params, net = extract_n_params(*feat, n_patch_dim)

    x_idx = np.digitize(n_hidden_layer, trial_n_hidden_layer) - 1
    y_idx = np.digitize(n_node_hidden, trial_n_node_hidden) - 1

    #for i_trial in range(n_trials):
    ds = np.load(fname)
    all_perf_cv[:, x_idx, y_idx] = ds['mses_cv']
    all_perf_tot[x_idx, y_idx] = ds['mse_tot'].item()
    all_n_params[x_idx, y_idx] = n_params

    # Extract and load net
    state_path = '{}/model_n_layer_{:d}_n_node_hidden_{:02d}_n_channel_04_rd_4.pkl'.format(dirname, n_hidden_layer, n_node_hidden)
    net.load_state_dict(torch.load(state_path, map_location='cpu'))

    pred = net(torch.tensor(feat_vec)).detach().numpy().squeeze()
    mse = np.mean((energies - pred)**2)

    print("n_hidden_layer: {}. n_node_hidden: {} mse: {:.2f}".format(n_hidden_layer, n_node_hidden, mse))

    all_perf_test[x_idx, y_idx] = mse

    all_nets[x_idx, y_idx] = net

avg_perf_cv = all_perf_cv.mean(axis=0)
min_perf_cv = all_perf_cv.min(axis=0)

plt.close('all')
fig, ax = plt.subplots()

norm = plt.Normalize(0, 40)
pc = ax.pcolormesh(xx, yy, all_perf_test, cmap='plasma_r', norm=norm)

ax.set_xticks(trial_n_hidden_layer)
ax.set_yticks(trial_n_node_hidden)
plt.colorbar(pc)

plt.show()


plt.close('all')
fig, ax = plt.subplots()
aic = 2*all_n_params + all_perf_tot

pc = ax.pcolormesh(xx, yy, aic, cmap='plasma_r')

ax.set_xticks(trial_n_hidden_layer)
ax.set_yticks(trial_n_node_hidden)
plt.colorbar(pc)

plt.show()

# Finally, fit linear and quadratic models to ko to compare
ko = ols_feat[:,0]
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(ko.reshape(-1,1), energies)
pred = reg.predict(ko.reshape(-1,1))

mse_linear = np.mean((energies-pred)**2)
aic_linear = 4 + mse_linear

reg_poly = linear_model.LinearRegression(fit_intercept=True)
poly_feat = np.vstack((ko, ko**2)).T
reg_poly.fit(poly_feat, energies)
pred = reg_poly.predict(poly_feat)

mse_quad = np.mean((energies-pred)**2)
aic_quad = 6 + mse_linear


idx_small = 0,0
idx_big = 2,8

mse_small = all_perf_test[idx_small]
mse_big = all_perf_test[idx_big]

n_params_small = all_n_params[idx_small]
n_params_big = all_n_params[idx_big]

indices = np.array([2,3,n_params_small,n_params_big])
mses = [mse_linear, mse_quad, mse_small, mse_big]

plt.close('all')
fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharey=True,figsize=(9,6))

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [cycle[0], cycle[1], 'k', 'k']

width = 1
ax1.bar(indices, mses, width=width, color=colors)
ax2.bar(indices, mses, width=width, color=colors)
ax3.bar(indices, mses, width=width, color=colors)

ax1.set_xlim(0, 5)
ax2.set_xlim(367,371)
ax3.set_xlim(4519, 4523)

ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)

ax3.yaxis.tick_right()

d = .025
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax2.plot((1-d, 1+d), (-d, +d), **kwargs)
ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax3.transAxes)
ax3.plot((-d, +d), (-d, +d), **kwargs)
ax3.plot((-d, +d), (1-d, 1+d), **kwargs)

ax1.set_xticks([2,3])
ax2.set_xticks([369])
ax3.set_xticks([4521])

plt.savefig("{}/Desktop/fig_model_comp".format(homedir), transparent=True)

