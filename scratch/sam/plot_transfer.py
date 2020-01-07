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

    print('{} by {}  ({} data points)'.format(P,Q,len(pt_indices)))

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

#    if P != Q:
#        continue


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


tot_feat_vec = np.vstack(list(all_feat_vec.values()))
d = tot_feat_vec - tot_feat_vec.mean(axis=0)
cov = np.dot(d.T, d) / d.shape[0]

tot_energies = np.concatenate(list(all_energies.values()))
#indices = np.array([0, 1, 3, 5, 8])


ener_06_06 = all_energies['energies_06_06']
ener_04_04 = all_energies['energies_04_04']
ener_04_09 = all_energies['energies_04_09']
ener_02_02 = all_energies['energies_02_02']

feat_06_06 = all_feat_vec['feat_06_06']
feat_04_04 = all_feat_vec['feat_04_04']
feat_04_09 = all_feat_vec['feat_04_09']
feat_02_02 = all_feat_vec['feat_02_02']

perf_mse_06_06, err, xvals, fit, reg_06_06 = fit_leave_one(feat_06_06[:,indices], ener_06_06)
perf_mse_04_04, err, xvals, fit, reg_04_04 = fit_leave_one(feat_04_04[:,indices], ener_04_04)
perf_mse_04_09, err, xvals, fit, reg_04_09 = fit_leave_one(feat_04_09[:,indices], ener_04_09)
perf_mse_02_02, err, xvals, fit, reg_02_02 = fit_leave_one(feat_02_02[:,indices], ener_02_02)


comb_coef = (reg_06_06.coef_ + reg_04_04.coef_ + reg_04_09.coef_)/3
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
non_int = np.dot(shape_feat_vec[:,indices], comb_coef)
ints = shape_energies - non_int

tmp = np.repeat(p_q, 2, axis=0)
feat = np.zeros_like(tmp)
feat[:,0] = tmp.prod(axis=1)
feat[:,1] = tmp.sum(axis=1)

# Fit intercepts to N, Next
perf_mse, err, xvals, fit, reg = fit_leave_one(feat, ints)
ax = plt.figure().gca(projection='3d')

nvals = np.arange(100)
nextvals = np.arange(40)

xx, yy = np.meshgrid(nvals, nextvals)

vals = reg.intercept_ + reg.coef_[0] * xx + reg.coef_[1] * yy

ax.plot_surface(xx, yy, vals, alpha=0.5)
#ax.scatter(feat[:,0], feat[:,1], ints)
ax.scatter(feat[::2,0], feat[::2,1], ints[1::2], label='methyl')
ax.scatter(feat[::2, 0], feat[::2,1], ints[::2], label='hydroxyl')

pred_phob = reg.predict(feat[::2])
pred_phil = reg.predict(feat[::2])

err_phob = ints[1::2] - pred_phob
err_phil = ints[::2] - pred_phil


