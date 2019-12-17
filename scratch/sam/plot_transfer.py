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
np.set_printoptions(precision=3)
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

# Gets rank of feat vec's cov matrix
#
def get_cov_rank(feat_vec):
    d = feat_vec - feat_vec.mean(axis=0)
    cov = np.dot(d.T, d) / d.shape[0]

    return np.linalg.matrix_rank(cov)


outdir = os.environ['HOME']
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


first_dirs = glob.glob('P*/k_*/PvN.dat')
assert len(first_dirs) % 2 == 0

other_dirs = ['P2/*/d_*/trial_*/PvN.dat', 'P3/k_00/d_*/trial_*/PvN.dat', 'P3/k_09/d_*/trial_*/PvN.dat', 'P4/*/d_*/trial_*/PvN.dat', 'P4_9/*/d_*/trial_*/PvN.dat']
all_dirs = first_dirs + other_dirs

for pathname in all_dirs:
    fnames = sorted(glob.glob(pathname))

    splits = fnames[0].split('/')
    size_list = np.array(splits[0][1:].split('_'), dtype=int)
    try:
        P, Q = size_list
    except ValueError:
        P, Q = size_list[0], size_list[0]

#    if P != Q:
#        continue


    print("Doing P: {}  Q: {}".format(P,Q))

    this_energies = np.zeros(len(fnames))
    pt_indices = []

    test_state = State(np.arange(P*Q).astype(int), P, Q)
    test_state.plot()
    plt.savefig('{}/Desktop/fig_{:02d}_{:02d}.png'.format(outdir, P, Q), transparent=True)
    plt.close('all')

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

    this_feat_vec, this_state = extract_feat(P, Q, pt_indices, this_energies)

    ener_key = 'energies_{:02d}_{:02d}'.format(P, Q)
    feat_key = 'feat_{:02d}_{:02d}'.format(P,Q)
    state_key = 'states_{:02d}_{:02d}'.format(P, Q)

    if ener_key in all_energies.keys():
        old_energies = all_energies[ener_key]
        old_feat_vec = all_feat_vec[feat_key]
        old_state = all_states[state_key]

        this_energies = np.append(old_energies, this_energies)
        this_feat_vec = np.vstack((old_feat_vec, this_feat_vec))
        this_state = np.append(old_state, this_state)

    all_energies[ener_key] = this_energies
    all_feat_vec[feat_key] = this_feat_vec
    all_states[state_key] = this_state


all_n_dat = np.array([energies.size for energies in all_energies.values()], dtype=int)
all_dat_keys = list(all_energies.keys())

names = np.array(['N', 'Next', 'k_c', 'k_o', 'n_mm', 'n_oo', 'n_mo', 'n_me', 'n_oe'])

tot_feat_vec = np.vstack([feat_vec for feat_vec in all_feat_vec.values()])
n_feat = tot_feat_vec.shape[1]


weights = np.zeros(all_n_dat.sum())
cum_n = np.cumsum(np.append(0, all_n_dat))
for i in range(all_n_dat.shape[0]):
    start = cum_n[i]
    end = cum_n[i+1]
    weights[start:end] = 1 / all_n_dat[i]

weights /= weights.sum()
weights = np.ones_like(weights)

tot_energies = np.concatenate([energy for energy in all_energies.values()])

# max num independent features
rank = get_cov_rank(tot_feat_vec)
for indices in itertools.combinations(np.arange(n_feat), rank):
    print("Doing {}".format(names[np.array(indices)]))
    this_rank = get_cov_rank(tot_feat_vec[:, indices])
    if this_rank < rank:
        continue

    tot_perf_r2, tot_perf_mse, tot_err, tot_xvals, tot_fit, tot_reg = fit_general_linear_model(tot_feat_vec[:,indices], tot_energies, sample_weight=weights)

    print("  perf: {:0.2f}".format(tot_perf_mse.mean()))
    print("  inter: {:0.2f}   coef: {}".format(tot_reg.intercept_, tot_reg.coef_))

indices = np.array([0, 1, 3, 5, 8])
indices2 = np.array([0, 1, 2, 4, 7])

tot_perf_r2, tot_perf_mse, tot_err, tot_xvals, tot_fit, tot_reg = fit_general_linear_model(tot_feat_vec[:, indices], tot_energies)

# Perform leave-one-out xval
for idx, name in enumerate(all_dat_keys):
    print("leaving out: {}".format(name))

    slc = slice(cum_n[idx], cum_n[idx+1], None)

    train_dat = np.delete(tot_feat_vec, slc, axis=0)
    train_y = np.delete(tot_energies, slc)

    test_dat = tot_feat_vec[slc]
    test_y = tot_energies[slc]
    assert np.unique(test_dat[:,0]).size == 1 and np.unique(test_dat[:,1]).size == 1

    reg = linear_model.LinearRegression()
    reg.fit(train_dat[:,indices], train_y)

    print('  Inter: {:.2f}   coef: {}'.format(reg.intercept_, reg.coef_))

    pred = reg.predict(test_dat[:,indices])

    perf_mse = np.mean((pred - test_y)**2).mean()
    print('  perf: {:.2f}'.format(perf_mse))

    r2 = 1 - perf_mse / test_y.var()
    print('  r2: {:.4f}'.format(r2))


