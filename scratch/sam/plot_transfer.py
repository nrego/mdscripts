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

    print('{} by {}  ({} data points)'.format(P,Q,len(pt_indices)))

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

headdirs = sorted(glob.glob('P*'))

# For each patch size ...
for headdir in headdirs:
    pathnames = sorted( glob.glob('{}/*/d_*/trial_*/PvN.dat'.format(headdir)) + glob.glob('{}/k_*/PvN.dat'.format(headdir)) )
    
    if len(pathnames) == 0:
        continue

    size_list = np.array(headdir[1:].split('_'), dtype=int)

    try:
        P, Q = size_list
    except ValueError:
        P, Q = size_list[0], size_list[0]

    print("Doing P: {}  Q: {}".format(P,Q))



    ## Collect all patterns and energies for all patches of this size
    this_energies = np.zeros(len(pathnames))
    pt_indices = []

    for i, fname in enumerate(pathnames):
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

tot_feat_vec = np.vstack(list(all_feat_vec.values()))
d = tot_feat_vec - tot_feat_vec.mean(axis=0)
cov = np.dot(d.T, d) / d.shape[0]

tot_energies = np.concatenate(list(all_energies.values()))
#indices = np.array([0, 1, 3, 5, 8])

#tot_perf_r2, tot_perf_mse, tot_err, tot_xvals, tot_fit, tot_reg = fit_general_linear_model(tot_feat_vec[:,indices], tot_energies)

indices = np.array([3, 5, 8])
# Just the k_00 and k_[P*Q]
shape_feat_vec = dict()
shape_energies = dict()
models = dict()

p_q = []

# Extract the shapes (k_00 and k_[p*q])
# and fit models
for e_key, f_key in zip(all_energies.keys(), all_feat_vec.keys()):

    p, q = np.array(e_key.split('_')[1:], dtype=int)
    p_q.append(np.array([p,q]))

    this_energies = all_energies[e_key]
    this_feat_vec = all_feat_vec[f_key]

    this_kvals = this_feat_vec[:,2]
    N, Next = this_feat_vec.astype(int)[0, :2]

    k0_idx = this_kvals == 0
    kn_idx = this_kvals == N
    #try:
    assert k0_idx.sum() == kn_idx.sum() == 1
    #except:
    #    continue
    idx = k0_idx | kn_idx
    print("e key: {}".format(e_key))
    print("  N: {:02d} Next: {:02d}".format(N, Next))

    shape_energies[e_key] = this_energies[idx]
    shape_feat_vec[f_key] = this_feat_vec[idx]

    if this_energies.size > 2:
        print(" fitting model...")
        perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(this_feat_vec[:,indices], this_energies)
        print('  Coef: {}'.format(reg.coef_))
        print('  Inter: {}'.format(reg.intercept_))
        models[e_key] = reg

p_q = np.array(p_q)

shape_energies = np.concatenate(list(shape_energies.values()))
shape_feat_vec = np.concatenate(list(shape_feat_vec.values()))

# Non-intercept values, fit on consensus regression coefs
non_int = np.dot(shape_feat_vec[:,indices], reg.coef_)
ints = shape_energies - non_int

# Fit intercepts to N, Next
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(shape_feat_vec[:,:2], ints, fit_intercept=True)

