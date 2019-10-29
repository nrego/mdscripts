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

def make_feat(methyl_mask, pos_ext):
    feat = np.zeros(pos_ext.shape[0])
    feat[patch_indices[methyl_mask]] = 1
    feat[patch_indices[~methyl_mask]] = -1

    return feat

def plot_feat(feat, ny=4, nz=4):
    this_feat = feat.reshape(ny, nz).T[::-1, :]

    return this_feat.reshape(1,1,ny,nz)


norm = plt.Normalize(-1,1)

## Collect all P2 patterns, compare with model 3

ds = np.load("m3.dat.npz")
#k_oh, n_mo_int, n_oo_ext
feat_vec = ds['feat_vec']
energies = ds['energies']
# k_oh, n_mo_int, n_mo_ext
coef = ds['reg_coef']
intercept = ds['reg_intercept']


fnames = sorted(glob.glob("k_*/d_*/trial_*/PvN.dat")) 


positions = gen_pos_grid(2)
pos_ext = gen_pos_grid(4, z_offset=True, shift_y=-1, shift_z=-1)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)
non_patch_indices = np.setdiff1d(np.arange(pos_ext.shape[0]), patch_indices)
nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)
# Locally indexed array: ext_count[i] gives number of non-patch neighbors to patch atom i
ext_count = np.zeros(4, dtype=int)
for i in range(4):
    ext_count[i] = np.intersect1d(non_patch_indices, nn_ext[i]).size


p2_energies = np.zeros(len(fnames))
p2_methyl_pos = np.zeros((len(fnames), 4))
p2_mo_int = np.zeros_like(p2_energies)
p2_oo_ext = np.zeros_like(p2_energies)

for i, fname in enumerate(fnames):
    dirname = os.path.dirname(fname)

    this_pt = np.loadtxt('{}/this_pt.dat'.format(dirname), ndmin=1).astype(int)

    this_pt_o = np.setdiff1d(np.arange(4), this_pt)
    methyl_mask = np.zeros(4, dtype=bool)
    methyl_mask[this_pt] = True

    this_energy = np.loadtxt(fname)[0,1]

    p2_energies[i] = this_energy
    p2_methyl_pos[i] = methyl_mask

    this_mo_int = 0
    this_oo_ext = 0

    for m_idx in this_pt:
        for n_idx in nn[m_idx]:
            this_mo_int += ~methyl_mask[n_idx]

    for o_idx in this_pt_o:
        this_oo_ext += ext_count[o_idx]

    p2_mo_int[i] = this_mo_int
    p2_oo_ext[i] = this_oo_ext




p2_koh = 4 - p2_methyl_pos.sum(axis=1)

p2_feat_vec = np.vstack((p2_koh, p2_mo_int, p2_oo_ext)).T
p2_perf_r2, p2_perf_mse, p2_err, p2_xvals, p2_fit, p2_reg = fit_general_linear_model(p2_feat_vec, p2_energies)
