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

from util import gen_pos_grid, construct_neighbor_dist_lists
from util import plot_pattern, plot_3d, plot_graph, plot_annotate, plot_edges
from util import gen_w_graph
from util import find_keff, find_keff_kernel
from util import fit_general_linear_model

import itertools

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 30})

### run from pooled_pattern_sample directory, after linking in ~/mdscripts/scratch/sam/util/ ###

ds = np.load('sam_pattern_data.dat.npz')

# 2d positions (in y, z) of all 36 patch head groups (For plotting schematic images, calculating order params, etc)
# Shape: (N=36, 2)
positions = ds['positions']

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
# i is list of indices to remove from pos ext to remove center patch
d, idx_to_remove = cKDTree(pos_ext).query(positions, k=1)
pos_ext = np.delete(pos_ext, idx_to_remove, axis=0)

# array of masks (boolean array) of methyl groups for each sample configuration
#  Shape: (n_samples=884, N=36)
methyl_pos = ds['methyl_pos']

# k_ch3 for each sample
k_vals = ds['k_vals']

# \beta F(0)'s
energies = ds['energies']

n_samples = methyl_pos.shape[0]
assert n_samples == energies.size == k_vals.size
nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)


## Compute k_eff for each sample ##

# edge types: methyl-methyl (mm), hydroxyl-hydoxyl (oo), methyl-hydroxyl (mo), methyl-extended (me), hydroxyl-extended (oe)
k_eff = np.zeros((n_samples, 5))
for i, methyl_mask in enumerate(methyl_pos):
    k_eff[i] = find_keff(methyl_mask, nn, nn_ext).sum(axis=0)

# Remove n_mo and n_oe, which are linear combo's of the other features
k_eff_all = k_eff
np.savetxt('k_eff_all.dat', k_eff_all)
k_eff = np.delete(k_eff_all, (1,4), axis=1)

perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(k_eff, energies)

methyl_mask = methyl_pos[500]
deg = find_keff(methyl_mask, nn, nn_ext)

k_eff_orig = np.zeros(n_samples)
for i, methyl_mask in enumerate(methyl_pos):
    wt_graph = gen_w_graph(positions, methyl_mask, wt_mm=1, wt_oo=0, wt_mo=0)
    deg = np.sum(dict(wt_graph.degree(weight='weight')).values())
    k_eff_orig[i] = deg

sig_sq = np.ones(5)
#rcut = np.ones(5) * 0.6
rcut = (0.6, -1, 1.2, 1.8, -1)

k_kernal = np.zeros((n_samples, 5))

for i, methyl_mask in enumerate(methyl_pos):
    k_kernal[i] = find_keff_kernel(methyl_mask, dd, dd_ext, sig_sq, rcut).sum(axis=0)

k_kernal = np.delete(k_kernal, (1,4), axis=1)

perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(k_kernal, energies)


methyl_mask = methyl_pos[500]

