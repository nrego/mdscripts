## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.neural_net.lib import *
from scratch.neural_net.run_pvn_simple import get_fit, xvals
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

n_out_channels = 4
n_hidden = 4
n_layers = 1

def count_n_mm(feat_vec, pos_ext):
    tree = cKDTree(pos_ext)
    pairs = tree.query_pairs(r=0.51)

    n_mm_tot = np.zeros(feat_vec.shape[0])

    for i_feat, feat in enumerate(feat_vec):
        n_mm = 0
        for i,j in pairs:
            if feat[i] == 1 and feat[j] == 1:
                n_mm += 1

        n_mm_tot[i_feat] = n_mm


    return n_mm_tot

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
def construct_pvn_images(idx, net, x_pattern, path='{}/Desktop'.format(home)):
    
    c, r, p = net.layer1.children()
    this_pattern = x_pattern[idx][None,:]

    # Apply conv filters to pattern
    out_all = r(c(x_pattern).detach())
    max0 = out_all[:,0].max()
    max1 = out_all[:,1].max()
    max2 = out_all[:,2].max()
    max3 = out_all[:,3].max()
    filter_norm = [Normalize(0,max0), Normalize(0,max1), Normalize(0,max2), Normalize(0,max3)]


    conv = r(c(this_pattern).detach())
    pool = p(conv)

    plot_hextensor(this_pattern, norm=Normalize(0,1))
    plt.savefig('{}/fnot_{:03}_pattern'.format(path, idx))
    plt.close('all')

    plot_hextensor(conv, cmap='Greys', norm=filter_norm)
    plt.savefig('{}/fnot_{:03d}_filter_conv'.format(path, idx))

    plot_hextensor(pool, cmap='Greys', norm=filter_norm)
    plt.savefig('{}/fnot_{:03d}_filter_pool'.format(path, idx))


home = os.environ['HOME']

from matplotlib.colors import Normalize
bphi_norm = Normalize(0.75, 2.0)

feat_vec, energies, poly, beta_phi_stars, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep()
emin, emax = energies.min(), energies.max()

aug_feat_vec, aug_energies = augment_data(feat_vec, energies)
dataset = SAMConvDataset(aug_feat_vec, aug_energies, norm_target=True, y_min=emin, y_max=emax)

k_c = aug_feat_vec.sum(axis=1)
n_mm = count_n_mm(aug_feat_vec, pos_ext)

fnames = glob.glob('model_*')

net = SAMConvNet(n_out_channels=n_out_channels, n_layers=n_layers, n_hidden=n_hidden, n_out=1)


for fname in fnames:
    state_dict = torch.load(fname)
    net.load_state_dict(state_dict)

    title = fname.split('.')[0].split('_')[-1]
    # Contains the conv layer, relu, and max pool
    l1 = net.layer1
    c, r, p = l1.children()

    k0 = c.kernel0
    k1 = c.kernel1

    arr = kernel_rep(k0, k1)
    plt.close('all')

    rng_pt = np.abs(arr[0].reshape(n_out_channels,9)).max(axis=1)

    norms = []
    #for i in range(n_out_channels):
    #    norms.append(plt.Normalize(-rng_pt[i], rng_pt[i]))
    norm = plt.Normalize(-rng_pt.max(), rng_pt.max())
    kernel_rep(k0, k1, norm=norm, cmap='RdBu')
    ax = plt.gca()
    #ax.set_title(title)

    plt.show()

x, y = dataset[:]
fname = fnames[0]
state_dict = torch.load(fname)
net.load_state_dict(state_dict)

construct_pvn_images(100, net, x)


