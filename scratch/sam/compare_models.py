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


## Compare models of different levels of sophistication - # params ##

ds = np.load('sam_pattern_data.dat.npz')

energies = ds['energies']
k_vals = ds['k_vals']
methyl_pos = ds['methyl_pos']
positions = ds['positions']

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
# patc_idx is list of patch indices
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

methyl_pos = ds['methyl_pos']
n_configs = methyl_pos.shape[0]

edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]

# shape: (n_samples, n_edges, n_conn_type)

# Conn types:   mm  oo  mo
k_eff_all_shape = np.load('k_eff_all.dat.npy')
assert n_edges == k_eff_all_shape.shape[1]


n_conn_type = k_eff_all_shape.shape[2]
## One model ##

# k_ch3 #
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(k_vals, energies, do_ridge=False)

# k_O #
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(36-k_vals, energies, do_ridge=False)


## n_mm n_oo  n_mo ##
conn_names = ['mm', 'oo', 'mo']

# Choose 2 to keep
indices = list(itertools.combinations(np.arange(n_conn_type), 2))
print("2 model")
for idx in indices:

    ## two edges ##
    
    feat_vec = k_eff_all_shape[:,:,idx].sum(axis=1)
    conn_0 = conn_names[idx[0]]
    conn_1 = conn_names[idx[1]]
    
    print("Model: {} {}".format(conn_0, conn_1))
    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

    print("  alpha_0: {:0.2f}".format(reg.intercept_))
    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[0]))
    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[1]))
    print("  perf: {:0.2f}".format(perf_mse.mean()))

    ## k_C and edge
    
    idx_incl = np.setdiff1d(np.arange(n_conn_type), np.array(idx))[0]
    feat_vec = np.delete(k_eff_all_shape, idx, axis=2).sum(axis=1)
    feat_vec = np.hstack((k_vals[:,None], feat_vec))
    conn_0 = conn_names[idx_incl]
    print("Model: {} {}".format("k_C", conn_0))
    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

    print("  alpha_0: {:0.2f}".format(reg.intercept_))
    print("  alpha_kC: {:0.2f}".format(reg.coef_[0]))
    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
    print("  perf: {:0.2f}".format(perf_mse.mean()))

    # k_O and edge
    
    idx_incl = np.setdiff1d(np.arange(n_conn_type), np.array(idx))[0]
    feat_vec = np.delete(k_eff_all_shape, idx, axis=2).sum(axis=1)
    feat_vec = np.hstack((36-k_vals[:,None], feat_vec))
    conn_0 = conn_names[idx_incl]
    print("Model: {} {}".format("k_O", conn_0))
    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

    print("  alpha_0: {:0.2f}".format(reg.intercept_))
    print("  alpha_kO: {:0.2f}".format(reg.coef_[0]))
    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
    print("  perf: {:0.2f}".format(perf_mse.mean()))


