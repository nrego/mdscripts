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

# Start with 6x6 patch
ds = np.load('sam_pattern_data.dat.npz')
energies66 = ds['energies']
methyl_pos = ds['methyl_pos']
pt_indices = []
for methyl_mask in methyl_pos:
    pt_indices.append(np.arange(36, dtype=int)[methyl_mask])

feat_vec66 = np.zeros((methyl_pos.shape[0], 9))
for i, pt_index in enumerate(pt_indices):
    state = State(pt_index, 6, 6)
    feat_vec66[i] = state.N_int, state.N_ext, state.k_c, state.k_o, state.n_mm, state.n_oo, state.n_mo, state.n_me, state.n_oe

fnames = sorted(glob.glob("*/*/d_*/trial_*/PvN.dat")) 

feat_vec44 = np.zeros((len(fnames), 9))
energies44 = np.zeros(len(fnames))
for i, fname in enumerate(fnames):
    dirname = os.path.dirname(fname)

    this_pt = np.loadtxt('{}/this_pt.dat'.format(dirname), ndmin=1).astype(int)
    patch = int(dirname.split('/')[0][1])
    state = State(this_pt, patch, patch)
    feat_vec44[i] = state.N_int, state.N_ext, state.k_c, state.k_o, state.n_mm, state.n_oo, state.n_mo, state.n_me, state.n_oe

    this_energy = np.loadtxt(fname)[0,1]
    this_err = np.loadtxt(fname)[0,2]
    energies44[i] = this_energy


# Hybrid model with different patch sizes??
# feat vec is:
#  N_int, N_ext, k_c, k_o, n_mm, n_oo, n_mo, n_me, n_oe
tot_feat_vec = np.vstack((feat_vec66, feat_vec44))

tot_energies = np.concatenate((energies66, energies44))

indices = np.array([1,2,5,6])
weights = np.ones(len(tot_energies))
#weights[-16:] = 884 / 16
weights /= weights.sum()

tot_perf_r2, tot_perf_mse, tot_err, tot_xvals, tot_fit, tot_reg = fit_general_linear_model(tot_feat_vec[:,indices], tot_energies, sample_weight=weights)

for k in np.arange(9, 0, -1):
    
    best_perf = np.inf
    best_indices = None

    for indices in combinations(np.arange(9), k):
        tot_perf_r2, tot_perf_mse, tot_err, tot_xvals, tot_fit, tot_reg = fit_general_linear_model(tot_feat_vec[:,indices], tot_energies)
        print("indices: {}   perf: {:.2f}".format(indices, tot_perf_mse.mean()))
        if tot_perf_mse.mean() < best_perf:
            best_perf = tot_perf_mse.mean()
            best_indices = np.array(indices)

    print("k: {:1d}  perf: {:.2f}  idx: {}".format(k, best_perf, best_indices))



