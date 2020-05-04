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
homedir=os.environ['HOME']
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':14})

## TODO: Move out of here

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

# Given a pattern, plot filters, filtering operations, and pooling
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
    k0, k1 = c.kernel0, c.kernel1
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
    conv2 = r(c(conv1).detach())
    pool2 = p(conv2).detach()

    plot_hextensor(conv2, cmap="Greys", norm=mynorm2)
    plt.savefig('{}/fig_{}_filter_conv2'.format(path, title), transparent=True)

    plot_hextensor(pool2, cmap="Greys", norm=mynorm2)
    plt.savefig('{}/fig_{}_filter_pool2'.format(path, title), transparent=True)

n_hidden_layer = 1
n_node_hidden = 2
n_conv_filters = 4

net = SAMConvNetSimple(n_conv_filters=n_conv_filters, n_hidden_layer=n_hidden_layer, n_node_hidden=n_node_hidden)

#state_dict = torch.load("model_n_hidden_layer_1_n_node_hidden_{:02d}_n_conv_filters_04_all.pkl".format(n_node_hidden), map_location='cpu')
state_dict = torch.load("model_n_layer_1_n_node_hidden_{:02d}_n_channel_{:02d}_rd_2.pkl".format(n_node_hidden, n_conv_filters), map_location='cpu')
net.load_state_dict(state_dict)

feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('sam_pattern_06_06.npz')
n_patch_dim = feat_vec.shape[1]
state = states[841]

dataset = SAMConvDataset(feat_vec, energies)

pred = net(dataset.X).detach().numpy().squeeze()
mse = np.mean((energies - pred)**2)

x = dataset.X[841]
x = x.reshape(-1,*x.shape)

## Plot conv kernels

norm = plt.Normalize(-1,1)

## conv layer 1

plt.close('all')
plot_hextensor(x)
plt.savefig('{}/Desktop/pattern_embed'.format(homedir), transparent=True)

l1 = net.conv1
c, r, p = l1.children()

k0 = c.kernel0
k1 = c.kernel1

arr = kernel_rep(k0, k1)
plt.close('all')


kernel_rep(k0, k1, norm=norm, cmap='RdBu')
plt.savefig('{}/Desktop/kernel_l1'.format(homedir), transparent=True)

## Conv layer 2

plt.close('all')
l2 = net.conv2
c, r, p = l2.children()

k0 = c.kernel0
k1 = c.kernel1

arr = kernel_rep(k0, k1)
plt.close('all')

kernel_rep(k0, k1, norm=norm, cmap='RdBu')
plt.savefig('{}/Desktop/kernel_l2'.format(homedir), transparent=True)

plt.close('all')

## Now, make all images

construct_pvn_images(841, net, dataset.X)

plt.close('all')

#l1 = net.conv1
#c, r, p = l1.children()
#out = r(c(x))
#plt.close('all')
