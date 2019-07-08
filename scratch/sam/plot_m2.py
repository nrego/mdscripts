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


k_eff_one_edge = k_eff_all_shape.sum(axis=1)

# K_C, n_mm
feat_vec = np.dstack((k_vals, k_eff_one_edge[:,0])).squeeze(axis=0)
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)
np.savetxt('m2_coef.dat', reg.coef_)
#np.savetxt('m2_int.dat', reg.intercept_)
norm = plt.Normalize(132, 287)
cmap = cm.Spectral

fig = plt.figure()
ax = fig.gca(projection='3d')

colors = cmap(norm(energies))

sc = ax.scatter(feat_vec[:,0], feat_vec[:,1], energies, c=colors)


fig = plt.figure()
ax = fig.gca()
sc = ax.scatter(feat_vec[:,0], feat_vec[:,1], c=energies, cmap=cmap, norm=norm)
plt.colorbar(sc)
ax.set_xticks(np.arange(0,42,6))
plt.savefig('{}/Desktop/m2.png'.format(homedir), transparent=True)


## Now print a random configuration to illustrate n_mm
edges, ext_indices = enumerate_edges(positions, [], nn, np.arange(36))
idx = 310
methyl_mask = methyl_pos[310]
methyl_indices = np.arange(36)[methyl_mask]

fig, ax = plt.subplots(figsize=(6,6))

ax.plot(positions[~methyl_mask, 0], positions[~methyl_mask, 1], 'bo', markersize=20)
ax.plot(positions[methyl_mask, 0], positions[methyl_mask, 1], 'ko', markersize=20)

for idx_i, idx_j in edges:
    if idx_i in range(36) and idx_j in range(36):
        pos_i = positions[idx_i]
        pos_j = positions[idx_j]
        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'k-', linewidth=4, zorder=0)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig('{}/Desktop/config.png'.format(homedir), transparent=True)


fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(k_vals, err, s=12, color='b')
#ax.plot(xvals, fit, 'k-', linewidth=3)
ax.set_xlabel(r'$k_C$')
ax.set_ylabel(r'$\hat{f} - f$')
fig.tight_layout()

ax.set_xticks(np.arange(0,42,6))
ax.set_ylim(-21,21)

fig.savefig('{}/Desktop/fit_two_err.png'.format(homedir), transparent=True)
plt.close('all')

