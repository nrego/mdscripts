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


## Plot CNN filters (first layer, possibly second layer, too)
##   For a cnn with a given set of hyper params (set below)
n_hidden_layer = 2
n_node_hidden = 4
n_conv_filters = 9

# Are there two convolutions?
is_double = True

homedir=os.environ['HOME']
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':14})

## TODO: Move out of here

## Need to initialize and load in states for CNNs manually...
def make_nets(all_state_dict, all_hyp_param_array):
    all_nets = np.empty_like(all_state_dict)

    d1, d2, d3 = all_state_dict.shape
    
    for (i_x, i_y, i_z) in itertools.product(np.arange(d1), np.arange(d2), np.arange(d3)):
        this_state_dict = all_state_dict[i_x,i_y,i_z]
        n_conv_filters, n_hidden_layer, n_node_hidden = all_hyp_param_array[i_x, i_y, i_z]

        net = SAMConvNet(n_conv_filters=n_conv_filters, n_hidden_layer=n_hidden_layer, 
                               n_node_hidden=n_node_hidden)
        net.load_state_dict(this_state_dict)

        all_nets[i_x, i_y, i_z] = net


    return all_nets 


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
def construct_pvn_images(idx, net, x_pattern, path='{}/Desktop'.format(homedir), title=None, is_double=False):

    c, r, p = net.conv1.children()
    this_pattern = x_pattern[idx][None,:]
    
    # Apply conv filters to all patterns to find max vals
    out_all = r(c(x_pattern).detach())
    max0 = out_all[:,0].max()

    mynorm1 = Normalize(0,max0)
    #filter_norm = [mynorm for i in range(out_all.shape[1])]


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
    if is_double:
        c, r, p = net.conv2.children()
        out_all = r(c(out_all).detach())
        max1 = out_all[:,0].max()

        mynorm2 = Normalize(0,max1)
        conv2 = r(c(pool1).detach())
        pool2 = p(conv2).detach()

        plot_hextensor(conv2, cmap="Greys", norm=mynorm2)
        plt.savefig('{}/fig_{}_filter_conv2'.format(path, title), transparent=True)

        plot_hextensor(pool2, cmap="Greys", norm=mynorm2)
        plt.savefig('{}/fig_{}_filter_pool2'.format(path, title), transparent=True)



##
ds = np.load('data/sam_cnn_ml_trials.npz')
all_state_dict = ds['all_state_dict']
# n_conv_filters, n_hidden_layer, n_node_hidden
all_hyp_param_array = ds['all_hyp_param_array']

all_nets = make_nets(all_state_dict, all_hyp_param_array)

trial_n_conv_filters = ds['trial_n_conv_filters']
trial_n_hidden_layer = ds['trial_n_hidden_layer']
trial_n_node_hidden = ds['trial_n_node_hidden']
n_sample = ds['n_sample'].item()

i_conv_filters = np.digitize(n_conv_filters, trial_n_conv_filters) - 1
i_hidden_layer = np.digitize(n_hidden_layer, trial_n_hidden_layer) - 1
i_node_hidden = np.digitize(n_node_hidden, trial_n_node_hidden) - 1


net = all_nets[i_conv_filters, i_hidden_layer, i_node_hidden]


#Get feat vec and augment to get right dimensions
feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('data/sam_pattern_06_06.npz')
n_patch_dim = feat_vec.shape[1]

homedir = os.environ['HOME']


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


kernel_rep(k0, k1, norm=norm, cmap='PiYG')
plt.savefig('{}/Desktop/kernel_l1'.format(homedir), transparent=True)


## Conv layer 2
if is_double:
    plt.close('all')
    l2 = net.conv2
    c, r, p = l2.children()

    k0 = c.kernel0
    k1 = c.kernel1

    arr = kernel_rep(k0, k1)
    plt.close('all')

    kernel_rep(k0, k1, norm=norm, cmap='PiYG')
    plt.savefig('{}/Desktop/kernel_l2'.format(homedir), transparent=True)

    plt.close('all')

## Now, make all images

construct_pvn_images(841, net, dataset.X, is_double=is_double)

plt.close('all')

## Plot sample conv ##
feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('data/sam_pattern_06_06.npz', embed_pos_ext=True, ny=8, nz=9)
dataset = SAMConvDataset(feat_vec, energies, ny=8, nz=9)
x = dataset.X[841][None, ...]

plot_hextensor(x, mask=np.arange(0,64,9))
plt.savefig('{}/Desktop/small_pattern'.format(homedir), transparent=True)
plt.close('all')

l1 = net.conv1
c, r, p = l1.children()
oc = r(c(x)).detach()[:,4,...][None,...]
plot_hextensor(oc, norm=plt.Normalize(0, 4.5), cmap='Oranges', mask=np.arange(0,64,9))
plt.savefig('{}/Desktop/small_pattern_conv'.format(homedir), transparent=True)

op = p(oc)

plt.close('all')

plot_hextensor(op, norm=plt.Normalize(0, 4.5), cmap='Oranges', )
plt.savefig('{}/Desktop/small_pattern_pool'.format(homedir), transparent=True)




