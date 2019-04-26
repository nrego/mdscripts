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

from util import *

import itertools

from sklearn.cluster import AgglomerativeClustering

from lib.wang_landau import WangLandau


def get_energy(methyl_mask, edges=None, patch_indices=None, reg=None):

    # n_mm_int, n_mo_int, n_mo_ext
    feat_vec = get_keff_all(methyl_mask, edges, patch_indices).sum(axis=0)[[0,2,3]]
    
    return np.dot(reg.coef_, feat_vec) + reg.intercept_

## Run WL algorithm on energies for each k_ch3
##   Energy function here is the 3 dof linear model (i.e. 3 connection types [mm,oo,mo], two edge types [int and ext])

## Load datasets to train model
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
int_mask = np.ones(n_edges, dtype=bool)
int_mask[ext_indices] = False

k_eff_all_shape = np.load('k_eff_all.dat.npy')
k_eff_int = k_eff_all_shape[:, int_mask, :].sum(axis=1)
k_eff_ext = k_eff_all_shape[:,~int_mask, :].sum(axis=1)

# n_mm_int, n_mo_int, n_mo_ext
feat_vec = np.hstack((k_eff_int[:, [0,2]], k_eff_ext[:, 2][:, None]))
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

bins = np.arange(135, 286, 1.)
fn_kwargs = dict(edges=edges, patch_indices=patch_indices, reg=reg)
wl = WangLandau(positions, bins, get_energy, fn_kwargs)
wl.gen_states(k=1, do_brute=True)
