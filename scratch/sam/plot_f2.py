from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import networkx as nx
from scipy.spatial import cKDTree

from sklearn import datasets, linear_model

from scipy.integrate import cumtrapz

from scratch.sam.util import *

import itertools

from sklearn.cluster import AgglomerativeClustering

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

homedir = os.environ['HOME']

def fit_model(data_vecs, indices, energies):
    reshaped_data = []
    for data_vec, idx in zip(data_vecs, indices):
        assert data_vec.ndim == 2
        new_vec = data_vec[:,idx]
        if new_vec.ndim == 1:
            reshaped_data.append(new_vec[:,None])
        elif new_vec.ndim == 2:
            reshaped_data.append(new_vec.copy())
        else:
            raise ValueError

    feat_vec = np.hstack(reshaped_data)

    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

    return reg

aics = []
perf_mses = []

## Compare models of different levels of sophistication - # params ##

ds = np.load('sam_pattern_data.dat.npz')

energies = ds['energies']
k_vals = ds['k_vals']
k_vals_both = np.hstack((k_vals[:,None], 36-k_vals[:,None]))
methyl_pos = ds['methyl_pos']
positions = ds['positions']

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
# patch_idx is list of patch indices in pos_ext 
#   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

# nn_ext is dictionary of (global) nearest neighbor's to each patch point
#   nn_ext[i]  global idxs of neighbor to local patch i 
nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

methyl_pos = ds['methyl_pos']
n_configs = methyl_pos.shape[0]

edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]


## k_eff_all_shape is an exhaustive list of the connection type of every
# edge for every pattern
## n_mm n_oo  n_mo ##
# Conn types:   mm  oo  mo
# shape: (n_samples, n_edges, n_conn_type)
k_eff_all_shape = np.load('k_eff_all.dat.npy')
assert n_edges == k_eff_all_shape.shape[1]
n_conn_type = k_eff_all_shape.shape[2]


## One model ##
###############

# k_ch3 #
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(36-k_vals, energies, do_ridge=False)

fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(36-k_vals, energies, s=12, color='b')
ax.plot(xvals, fit, 'k-', linewidth=3)
#ax.set_xlabel(r'$k_C$')
#ax.set_ylabel(r'$f$')
fig.tight_layout()

ax.set_xticks(np.arange(0,42,6))

fig.savefig('{}/Desktop/fit_one.png'.format(homedir), transparent=True)
plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(36-k_vals, err, s=12, color='b')
#ax.plot(xvals, fit, 'k-', linewidth=3)
#ax.set_xlabel(r'$k_C$')
#ax.set_ylabel(r'$\hat{f} - f$')
fig.tight_layout()

ax.set_xticks(np.arange(0,42,6))
ax.set_ylim(-21,21)

fig.savefig('{}/Desktop/fit_one_err.png'.format(homedir), transparent=True)
plt.close('all')


