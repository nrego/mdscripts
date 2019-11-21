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
from itertools import combinations

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
    feat_vec66[i] = 6, 6, state.k_c, state.k_o, state.n_mm, state.n_oo, state.n_mo, state.n_me, state.n_oe



fnames = sorted(glob.glob("P4/*/d_*/trial_*/PvN.dat")) 

feat_vec44 = np.zeros((len(fnames), 9))
energies44 = np.zeros(len(fnames))
for i, fname in enumerate(fnames):
    dirname = os.path.dirname(fname)

    this_pt = np.loadtxt('{}/this_pt.dat'.format(dirname), ndmin=1).astype(int)
    patch = int(dirname.split('/')[0][1])
    state = State(this_pt, patch, patch)
    feat_vec44[i] = patch, patch, state.k_c, state.k_o, state.n_mm, state.n_oo, state.n_mo, state.n_me, state.n_oe

    this_energy = np.loadtxt(fname)[0,1]
    this_err = np.loadtxt(fname)[0,2]
    energies44[i] = this_energy

    if i % 20 == 0:
        state.plot()
        plt.savefig('/Users/nickrego/Desktop/fig_{:02d}'.format(i))
        plt.close('all')


fnames = sorted(glob.glob("P2/*/d_*/trial_*/PvN.dat")) 

feat_vec22 = np.zeros((len(fnames), 9))
energies22 = np.zeros(len(fnames))
for i, fname in enumerate(fnames):
    dirname = os.path.dirname(fname)

    this_pt = np.loadtxt('{}/this_pt.dat'.format(dirname), ndmin=1).astype(int)
    patch = int(dirname.split('/')[0][1])
    state = State(this_pt, patch, patch)
    feat_vec22[i] = patch, patch, state.k_c, state.k_o, state.n_mm, state.n_oo, state.n_mo, state.n_me, state.n_oe

    this_energy = np.loadtxt(fname)[0,1]
    this_err = np.loadtxt(fname)[0,2]
    energies22[i] = this_energy


names = np.array(['P', 'Q', 'k_c', 'k_o', 'n_mm', 'n_oo', 'n_mo', 'n_me', 'n_oe'])

# Hybrid model with different patch sizes??
# feat vec is:
#  N_int, N_ext, k_c, k_o, n_mm, n_oo, n_mo, n_me, n_oe
tot_feat_vec = np.vstack((feat_vec44, feat_vec22))

tot_energies = np.concatenate((energies44, energies22))

indices = np.array([0,3,5,8])

#tot_feat_vec = feat_vec22
#tot_energies = energies22

weights = np.ones(len(tot_energies))
weights[:226] = 1/226
weights[226:226+16] = 1/16
#weights[884+226:] = 1/16
weights /= weights.sum()

#weights = np.ones_like(tot_energies)

tot_perf_r2, tot_perf_mse, tot_err, tot_xvals, tot_fit, tot_reg = fit_general_linear_model(tot_feat_vec[:,indices], tot_energies, sample_weight=weights)


'''
names = np.array(['P', 'Q', 'k_c', 'k_o', 'n_mm', 'n_oo', 'n_mo', 'n_me', 'n_oe'])

names = names[2:]
tot_energies = energies66
tot_feat_vec = feat_vec66[:,2:]

d = tot_feat_vec - tot_feat_vec.mean(axis=0)
tot_cov = np.dot(d.T, d) / d.shape[0]

for k in np.arange(tot_feat_vec.shape[1], 0, -1):
    
    cnt = 0
    for indices in combinations(np.arange(tot_feat_vec.shape[1]), k):

        this_feat = tot_feat_vec[:,indices]
        d = this_feat - this_feat.mean(axis=0)

        cov = np.dot(d.T, d) / d.shape[0]
        if np.linalg.matrix_rank(cov) < cov.shape[0]:
            continue
        
        cnt += 1
        tot_perf_r2, tot_perf_mse, tot_err, tot_xvals, tot_fit, tot_reg = fit_general_linear_model(this_feat, tot_energies)
        print("{}   perf: {:.2f}".format(names[np.array(indices)], tot_perf_mse.mean()))
        print("  {}   {:.2f}".format(tot_reg.coef_, tot_reg.intercept_))
'''
