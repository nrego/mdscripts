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
int_indices = np.setdiff1d(np.arange(n_edges), ext_indices)

## k_eff_all_shape is an exhaustive list of the connection type of every
# edge for every pattern
## n_mm n_oo  n_mo ##
# Conn types:   mm  oo  mo
# shape: (n_samples, n_edges, n_conn_type)
k_eff_all_shape = np.load('k_eff_all.dat.npy')
assert n_edges == k_eff_all_shape.shape[1]
n_conn_type = k_eff_all_shape.shape[2]


k_eff_int_edge = k_eff_all_shape[:, int_indices, :].sum(axis=1)
k_eff_ext_edge = k_eff_all_shape[:, ext_indices, :].sum(axis=1)

# k_oh, n_mo_int, n_oo_ext
#feat_vec = np.dstack((k_eff_int_edge[:,0], k_eff_int_edge[:,2], k_eff_ext_edge[:,2])).squeeze(axis=0)
feat_vec = np.dstack((36-k_vals, k_eff_int_edge[:,1], k_eff_ext_edge[:,1])).squeeze(axis=0)
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

norm = plt.Normalize(132, 287)
cmap = cm.Spectral

fig = plt.figure()
ax = fig.gca(projection='3d')

colors = cmap(norm(energies))

sc = ax.scatter(feat_vec[:,0], feat_vec[:,1], feat_vec[:,2], c=colors)


fig = plt.figure()
ax = fig.gca()
sc = ax.scatter(feat_vec[:,0], feat_vec[:,1], c=energies, cmap=cmap, norm=norm)
plt.colorbar(sc)
ax.set_xticks(np.arange(0,42,6))
plt.savefig('{}/Desktop/m3.png'.format(homedir), transparent=True)


## Now print a random configuration to illustrate n_mm_int, n_mo_int, n_mo_ext
idx = 310
methyl_mask = methyl_pos[310]

methyl_indices_local = np.arange(36)[methyl_mask]
methyl_indices_global = patch_indices[methyl_mask]
hydroxyl_indices_global = np.setdiff1d(patch_indices, methyl_indices_global)
edge_indices_global = np.setdiff1d(np.unique(edges[ext_indices]), patch_indices)

fig, ax = plt.subplots(figsize=(6,6))

ax.plot(positions[~methyl_mask, 0], positions[~methyl_mask, 1], 'bo', markersize=20, zorder=3)
ax.plot(positions[methyl_mask, 0], positions[methyl_mask, 1], 'ko', markersize=20, zorder=3)
ax.plot(pos_ext[edge_indices_global,0], pos_ext[edge_indices_global,1], 'bx', markersize=20, zorder=3)

for idx_i, idx_j in edges:
    # Skip oo_int and oo_ext
    if idx_i not in methyl_indices_global and idx_j not in methyl_indices_global:
        continue
    # mm edge
    if idx_i in methyl_indices_global and idx_j in methyl_indices_global:
        pos_i = pos_ext[idx_i]
        pos_j = pos_ext[idx_j]
        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'k-', linewidth=4)
    # mo ext edge
    elif idx_i in edge_indices_global or idx_j in edge_indices_global:
        pos_i = pos_ext[idx_i]
        pos_j = pos_ext[idx_j]
        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'b:', linewidth=4)
    # Must be mo int edge 
    else:       
        pos_i = pos_ext[idx_i]
        pos_j = pos_ext[idx_j]
        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 'b--', linewidth=4)

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

fig.savefig('{}/Desktop/fit_three_err.png'.format(homedir), transparent=True)
plt.close('all')

np.savez_compressed('m3.dat', feat_vec=feat_vec, energies=energies, methyl_pos=methyl_pos, reg_coef=reg.coef_, reg_intercept=reg.intercept_)


