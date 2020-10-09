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


## Plot patterns, illustrating rotational invariance
#
#
n_hidden_layer = 2
n_node_hidden = 4
n_conv_filters = 10

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
feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('data/sam_pattern_06_06.npz', binary_encoding=True)
feat_vec2, _,_,_,_,_ = load_and_prep('data/sam_pattern_06_06.npz', binary_encoding=False)


feat_vec, energies = hex_augment_data(feat_vec, energies, pos_ext, patch_indices, binary_encoding=False)

n_patch_dim = feat_vec.shape[1]

homedir = os.environ['HOME']


dataset = SAMConvDataset(feat_vec, energies)

X = dataset.X.reshape(dataset.X.shape[0], 1, *dataset.X.shape[1:])


idx = 841

x_rot = np.zeros((6, *X.shape[1:]))

# Collect each rotated image x
for i in range(6):
#for i in [0, 2, 4]:

    x_rot[i] = X[idx*6 + i]

    plt.close('all')
    plot_hextensor(x_rot[i])
    plt.savefig('{}/Desktop/pattern_embed_{:d}'.format(homedir, i), transparent=True)


    construct_pvn_images(idx*6+i, net, dataset.X, is_double=True, title=i)

plt.close('all')
cust_color = (1,0.5,0,0.6)
cust_color = (0.5,0.5,0.5,0.5)
arr = np.zeros((1,1,3,3))
#lw = np.ones(6)*6
#lw[3] = 12
plot_hextensor(arr, cust_color=cust_color, mask=(0,6))
plt.savefig('{}/Desktop/conv_filter'.format(homedir), transparent=True)

state = states[idx]
plt.close('all')
state.plot()
plt.savefig('{}/Desktop/pattern'.format(homedir), transparent=True)


### Plot first convolutional filter ###

plt.close('all')

c, r, p = net.conv1.children()

x = plot_feat(make_feat(state.methyl_mask, state.pos_ext, state.patch_indices))
x = torch.from_numpy(np.ascontiguousarray(x)).float()
c_out = c(x).detach()[:,0,...].reshape(1,1,8,8)

norm = plt.Normalize(0, np.ceil(c_out.max()))
plot_hextensor(c_out, cmap="Greys", norm=norm)
plt.savefig('{}/Desktop/conv'.format(homedir), transparent=True)
plt.close('all')


arr = np.zeros((1,1,1,1))
plot_hextensor(arr, cust_color=(0,0,0,0), linewidth=30)
plt.savefig('{}/Desktop/hex_fig'.format(homedir), transparent=True)

plt.close('all')


## Now for pooling
plt.close('all')
cust_color = (0.5,1.0,0,0.2)
cust_color = (0.5,0.5,0.5,0)
arr = np.zeros((1,1,3,3))
lw = np.ones(6)*12
#lw[3] = 12
plot_hextensor(arr, cust_color=cust_color, mask=(0,6), linewidth=lw)
plt.savefig('{}/Desktop/pool_filter.svg'.format(homedir), transparent=True)

##
plt.close('all')

arr = np.random.rand(1,1,3,3)
arr[0,0,1,2] = 10
lw = np.ones(6)*4
#lw[3] = 12
plot_hextensor(arr, norm=plt.Normalize(0,4.5), cmap='Oranges', linewidth=lw, mask=(0,6))
plt.savefig('{}/Desktop/pool_illustration'.format(homedir), transparent=True)

plot_hextensor(arr, norm=plt.Normalize(0,4.5), cmap='Oranges', linewidth=lw, mask=np.delete(np.arange(9),7))


plt.close('all')
x = make_feat(state.methyl_mask, state.pos_ext, state.patch_indices)
x = np.append(np.zeros(8), x)
x = plot_feat(x, 9, 8)
x = torch.from_numpy(np.ascontiguousarray(x)).float()
c_out = c(x).detach()[:,0,...].reshape(1,1,9,8)

p_out = p(r(c_out))
norm = plt.Normalize(0, np.ceil(p_out.max()))
plot_hextensor(p_out, cmap="Greys", norm=norm)
plt.savefig('{}/Desktop/pool'.format(homedir), transparent=True)
plt.close('all')


