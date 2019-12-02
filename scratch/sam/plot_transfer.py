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

### Add small plate datasets to build/test transferable models ###
# Run from small_patterns/ directory

# Create a feature vector and energy vector for a given patch size (given by p,q)
#   and associated patterns (pt_indices, local patch indexing) and energies
def extract_feat(P, Q, pt_indices, energies):
    N = P*Q
    # perimeter
    Nout = 2*(P+Q) - 4

    feat_vec = np.zeros((len(pt_indices), 9))
    states = np.empty(len(pt_indices), dtype=object)

    for i, pt_index in enumerate(pt_indices):
        state = State(pt_index, P, Q)
        states[i] = state
        feat_vec[i] = N, Nout, state.k_c, state.k_o, state.n_mm, state.n_oo, state.n_mo, state.n_me, state.n_oe


    return feat_vec, states


all_feat_vec = dict()
all_energies = dict()
all_states = dict()

# Start with 6x6 patch
ds = np.load('sam_pattern_data.dat.npz')
energies66 = ds['energies']
methyl_pos = ds['methyl_pos']
pt_indices = []
for methyl_mask in methyl_pos:
    pt_indices.append(np.arange(36, dtype=int)[methyl_mask])

all_feat_vec['feat_06_06'], all_states['states_06_06'] = extract_feat(6, 6, pt_indices, energies66)
all_energies['energies_06_06'] = energies66



other_dirs = ['P2/*/d_*/trial_*/PvN.dat', 'P2_08/k_*/PvN.dat', 'P2_12/k_*/PvN.dat', 'P2_18/k_*/PvN.dat', 'P3/*/d_*/trial_*/PvN.dat', 'P4/*/d_*/trial_*/PvN.dat', 'P4_9/*/d_*/trial_*/PvN.dat']


for pathname in other_dirs:
    fnames = sorted(glob.glob(pathname))

    splits = fnames[0].split('/')
    size_list = np.array(splits[0][1:].split('_'), dtype=int)
    try:
        P, Q = size_list
    except ValueError:
        P, Q = size_list[0], size_list[0]

    print("Doing P: {}  Q: {}".format(P,Q))

    this_energies = np.zeros(len(fnames))
    pt_indices = []

    for i, fname in enumerate(fnames):
        this_energies[i] = np.loadtxt(fname)[0,1]
        kc = int(fname.split('/')[1].split('_')[1])

        if kc == 0:
            this_pt = np.array([], dtype=int)
        elif kc == P*Q:
            this_pt = np.arange(P*Q, dtype=int)
        else:
            dirname = os.path.dirname(fname)
            this_pt = np.loadtxt('{}/this_pt.dat'.format(dirname), ndmin=1, dtype=int)

        pt_indices.append(this_pt)

    this_feat_vec = extract_feat(P, Q, pt_indices, this_energies)

    all_feat_vec['feat_{:02d}_{:02d}'.format(P, Q)], all_states['states_{:02d}_{:02d}'.format(P, Q)] = this_feat_vec
    all_energies['energies_{:02d}_{:02d}'.format(P, Q)] = this_energies


names = np.array(['N', 'Next', 'k_c', 'k_o', 'n_mm', 'n_oo', 'n_mo', 'n_me', 'n_oe'])

tot_feat_vec = np.vstack((all_feat_vec['feat_06_06'], all_feat_vec['feat_04_04']))
d = tot_feat_vec - tot_feat_vec.mean(axis=0)
cov = np.dot(d.T, d) / d.shape[0]

tot_energies = np.append(all_energies['energies_06_06'], all_energies['energies_04_04'])
indices = np.array([0, 3, 5, 8])

tot_perf_r2, tot_perf_mse, tot_err, tot_xvals, tot_fit, tot_reg = fit_general_linear_model(tot_feat_vec[:,indices], tot_energies)



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
