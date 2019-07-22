## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.neural_net.lib import *
from scratch.neural_net.run_pvn_simple import get_fit, xvals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import argparse
import os

home = os.environ['HOME']

from matplotlib.colors import Normalize
bphi_norm = Normalize(0.75, 2.0)

feat_vec, energies, poly, beta_phi_stars, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep()

p_min = poly.min(axis=0).astype(np.float32)
p_max = poly.max(axis=0).astype(np.float32)
p_range = p_max - p_min
p_range_mat = np.diag(p_range)

def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(figsize=(6, 1), subplot_kw=dict(xticks=[], yticks=[]))
    ax.imshow([colors], extent=[0, 10, 0, 1], alpha=0.9)


# Reshape an (n x 36) array to a (N, 1, 6, 6) using hexagdly's addressing scheme
def reshape_to_pic(arr):

    arr = np.array(arr)
    n_pts = arr.shape[0]
    img_shape = np.zeros((n_pts, 6, 6))

    for i in range(n_pts):
        this_feat = arr[i]
        this_feat = this_feat.reshape(6,6).T[::-1, ::-1]

        img_shape[i] = this_feat

    return torch.tensor(img_shape.astype(np.float32))

# Given a pattern, save its dewetting order as well as 
#   its predicted. Then, save CNN filters for pattern
def construct_beta_phi_images(idx, net, x_pattern, x_beta_phi_star, path='{}/Desktop'.format(home)):
    
    c, r, p = net.layer1.children()
    this_pattern = x_pattern[idx][None,:]

    out_all = r(c(x_pattern).detach())
    max0 = out_all[:,0].max()
    max1 = out_all[:,1].max()
    max2 = out_all[:,2].max()
    max3 = out_all[:,3].max()
    filter_norm = [Normalize(0,max0), Normalize(0,max1), Normalize(0,max2), Normalize(0,max3)]


    this_dewet = x_beta_phi_star[idx][None,:]
    this_pred = reshape_to_pic(net(this_pattern).detach())[None,:]

    act_pred = torch.cat((this_dewet, this_pred, this_pattern), dim=1)

    conv = r(c(this_pattern).detach())
    pool = p(conv)

    plot_hextensor(act_pred, norm=[bphi_norm, bphi_norm, Normalize(0,1)],
                   cmap=['bwr_r', 'bwr_r', mymap])
    plt.savefig('{}/bphi_{:03d}_pattern'.format(path, idx))

    plot_hextensor(conv, cmap='Greys', norm=filter_norm)
    plt.savefig('{}/bphi_{:03d}_filter_conv'.format(path, idx))

    plot_hextensor(pool, cmap='Greys', norm=filter_norm)
    plt.savefig('{}/bphi_{:03d}_filter_pool'.format(path, idx))

# Given a pattern, save its PvN as well as predicted
#  Save CNN filters for pattern
def construct_pvn_images(idx, net, x_pattern, y_pattern, path='{}/Desktop'.format(home)):
    
    c, r, p = net.layer1.children()
    this_pattern = x_pattern[idx][None,:]
    this_act = y_pattern[idx]

    this_pred = net(this_pattern).detach()[0]

    nvals, act = get_fit(this_act, p_min, p_range_mat, xvals)
    _, pred = get_fit(this_pred, p_min, p_range_mat, xvals)

    conv = r(c(this_pattern).detach())
    pool = p(conv)

    plot_hextensor(this_pattern)
    plt.savefig('{}/pvn_{:03}_pattern'.format(path, idx))
    plt.plot(nvals, act)
    plt.plot(nvals, pred, 'k--')
    plt.savefig('{}/pvn_{:03}_pred_act'.format(path, idx))

    plot_hextensor(conv, cmap='Greys', norm=filter_norm)
    plt.savefig('{}/pvn_{:03d}_filter_conv'.format(path, idx))

    plot_hextensor(pool, cmap='Greys', norm=filter_norm)
    plt.savefig('{}/pvn_{:03d}_filter_pool'.format(path, idx))


dataset_pattern = SAMConvDataset(feat_vec, poly, norm_target=True, y_min=p_min, y_max=p_max)

pvn_dict = torch.load("pvn_net.pkl")
net_pvn = SAMConvNet(n_hidden=18, n_out=5)
net_pvn.load_state_dict(pvn_dict)

beta_phi_dict = torch.load("beta_phi_net.pkl")
net_beta_phi = SAMConvNet(n_hidden=18, n_out=36)
net_beta_phi.load_state_dict(beta_phi_dict)

dataset_beta_phi_star = SAMConvDataset(beta_phi_stars, poly)

x_pattern, y_pattern = dataset_pattern[:]
x_beta_phi_star, y_beta_phi_star = dataset_beta_phi_star[:]