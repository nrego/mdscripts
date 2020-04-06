## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.neural_net.lib import *
from scratch.neural_net.run_pvn_simple import get_fit, xvals
from scratch.sam.util import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from scipy.spatial import cKDTree

import argparse
import os, glob
from matplotlib.colors import Normalize

home = os.environ['HOME']
## Hyper params


n_conv_filters = 6
n_hidden_layer = 2
n_node_hidden = 12

#headdir = '{}/n_layer_{:1d}/n_filter_{:02d}'.format(maindir, n_hidden_layer, n_conv_filters)
headdir = '.'

pattern_idx = 700
#pattern_idx = 8401
#pattern_idx = 10000

def kernel_rep(k0, k1, norm=None, cmap=None):
    k0 = k0.detach()[:,0,...]
    k1 = k1.detach()[:,0,...]

    n_filters = k0.shape[0]
    assert n_filters == k1.shape[0]

    arr = np.zeros((n_filters, 3, 3))

    for i_filter in range(n_filters):
        this_k0 = k0[i_filter]
        this_k1 = k1[i_filter]

        arr[i_filter][1:, 0] = this_k1[:,0]
        arr[i_filter][1:, 2] = this_k1[:,1]
        arr[i_filter][:, 1] = this_k0[:,0]

    arr = arr.reshape(1, n_filters, 3, 3)

    plot_hextensor(arr, norm=norm, cmap=cmap, mask=[0,6])

    return arr


# Given a pattern, save its PvN as well as predicted
#  Save CNN filters for pattern
def construct_pvn_images(idx, net, x_pattern, path='{}/Desktop'.format(homedir), title=None):

    c, r, p = net.conv1.children()
    this_pattern = x_pattern[idx][None,:]
    
    # Apply conv filters to all patterns to find max vals
    out_all = r(c(x_pattern).detach())
    max0 = out_all[:,0].max()

    mynorm1 = Normalize(0,max0)
    #filter_norm = [mynorm for i in range(out_all.shape[1])]

    c, r, p = net.conv2.children()
    out_all = r(c(out_all).detach())
    max1 = out_all[:,0].max()

    mynorm2 = Normalize(0,max1)

    # Convolve and max-pool this particular pattern at index idx
    c, r, p = net.conv1.children()
    conv1 = r(c(this_pattern).detach())
    pool1 = p(conv1).detach()

    plot_hextensor(this_pattern, norm=Normalize(-1,1))
    plt.savefig('{}/fig_idx_{}_pattern'.format(path, title), transparent=True)
    plt.close('all')

    #plot_hextensor(conv, cmap='Greys', norm=filter_norm)
    plot_hextensor(conv1, cmap="Greys", norm=mynorm1)
    plt.savefig('{}/fig_{}_filter_conv1'.format(path, title), transparent=True)

    plot_hextensor(pool1, cmap='Greys', norm=mynorm1)
    plt.savefig('{}/fig_{}_filter_pool1'.format(path, title), transparent=True)

    ## Now do layer 2
    c, r, p = net.conv2.children()
    conv2 = r(c(pool1).detach())
    pool2 = p(conv2).detach()

    plot_hextensor(conv2, cmap="Greys", norm=mynorm2)
    plt.savefig('{}/fig_{}_filter_conv2'.format(path, title), transparent=True)

    plot_hextensor(pool2, cmap="Greys", norm=mynorm2)
    plt.savefig('{}/fig_{}_filter_pool2'.format(path, title), transparent=True)



homedir = os.environ['HOME']

from matplotlib.colors import Normalize


feat_vec, patch_indices, pos_ext, energies, delta_e, dg_bind, weights, ols_feat, states = load_and_prep()
feat_idx = np.array([2,3,4])

y = dg_bind
emin = y.min()
emax = y.max()
n_dat = y.size

aug_feat_vec, aug_y = hex_augment_data(feat_vec, y, pos_ext, patch_indices)

#dataset = SAMConvDataset(aug_feat_vec, aug_y, norm_target=True, y_min=emin, y_max=emax)
dataset = SAMConvDataset(feat_vec, y, norm_target=False)
x, y = dataset[:]

k_c = (aug_feat_vec == 1).sum(axis=1)
k_o = (aug_feat_vec == -1).sum(axis=1)

net = SAMConvNet(n_conv_filters=n_conv_filters, n_hidden_layer=n_hidden_layer, n_node_hidden=n_node_hidden, n_out=1)


fnamemodel = '{}/model_n_hidden_layer_{:1d}_n_node_hidden_{:02d}_n_conv_filters_{:02d}_all.pkl'.format(headdir, n_hidden_layer, n_node_hidden, n_conv_filters)
fnameperf = '{}/perf_model_n_hidden_layer_{:1d}_n_node_hidden_{:02d}_n_conv_filters_{:02d}.npz'.format(headdir, n_hidden_layer, n_node_hidden, n_conv_filters)

state_dict = torch.load(fnamemodel, map_location="cpu")
net.load_state_dict(state_dict)

title = os.path.dirname(fnamemodel)
print("Filters: {}".format(title))

perf = np.load(fnameperf)['mses_cv'].mean()
tot_perf = np.load(fnameperf)['mse_tot']
print("Mean MSE: {:0.2f}".format(perf))
print("Tot MSE: {:0.2f}".format(tot_perf))


# Contains the conv layer, relu, and max pool
l1 = net.conv1
c, r, p = l1.children()

k0 = c.kernel0
k1 = c.kernel1

arr = kernel_rep(k0, k1)
plt.close('all')

# Get largest weight for each of the filters (channels)
rng_pt = np.abs(arr[0].reshape(n_conv_filters,9)).max(axis=1)

norms = []

norm = plt.Normalize(-rng_pt.max(), rng_pt.max())
print('max kernel weight: {:.2f}'.format(rng_pt.max()))


kernel_rep(k0, k1, norm=norm, cmap='RdBu')
ax = plt.gca()

plt.savefig('{}/Desktop/cnn_filter1.png'.format(homedir), transparent=True)

### Now for filters in layer 2
l2 = net.conv2
c, r, p = l2.children()

k0 = c.kernel0
k1 = c.kernel1

arr = kernel_rep(k0, k1)
plt.close('all')

# Get largest weight for each of the filters (channels)
rng_pt = np.abs(arr[0].reshape(n_conv_filters,9)).max(axis=1)

norms = []

norm = plt.Normalize(-rng_pt.max(), rng_pt.max())
print('max kernel weight: {:.2f}'.format(rng_pt.max()))


kernel_rep(k0, k1, norm=norm, cmap='RdBu')
ax = plt.gca()

plt.savefig('{}/Desktop/cnn_filter2'.format(homedir), transparent=True)




title = 'idx_{}'.format(pattern_idx)
construct_pvn_images(pattern_idx, net, x, title=title)

k_o = (feat_vec == -1).sum(axis=1)
pred = net(x[:884]).detach().numpy().squeeze()
err = dg_bind[:884] - pred


