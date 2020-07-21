
import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

from scipy.special import binom

#### RUN after 'extract_dos' (and after moving sam_dos to sam_data/data)
###
### This routine extracts the actual energies from the dos (since the dos is a fn of ko,noo,noe)
#
plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})

n_trials = 1000

### EXtract independent greedy design trials
#.   Run from ~/sam_design_trials

def extract_p_q(fname):
    splits = fname.split('_')

    p = int(splits[2])
    q = int(splits[4])

    return p,q

def get_feat(x, adj_mat, ext_count):
    xprime = 1-x

    ko = xprime.sum()
    noo = int(0.5*np.linalg.multi_dot((xprime, adj_mat, xprime)))
    noe = np.dot(ext_count, xprime)

    return (ko, noo, noe)

fnames = sorted(glob.glob('break_p*idx_000.npy'))
reg = np.load('sam_reg_m3.npy').item()

alpha1, alpha2, alpha3 = reg.coef_
alpha_n_cc = alpha2
alpha_n_ce = alpha2 - alpha3

all_states_break = dict()
all_states_build = dict()

all_f_break = dict()
all_f_build = dict()

for fname in fnames:
    p, q = extract_p_q(fname)
    print("doing p: {} q: {}".format(p,q))
    n = p*q
    break_parts = sorted(glob.glob('break_p_{:02d}_q_{:02d}_idx*.npy'.format(p,q)))

    state_np = State(np.arange(n), p=p, q=q)
    state_po = State(np.array([], dtype=int), p=p, q=q)

    this_max_delta_f = np.dot(np.array([state_po.k_o, state_po.n_oo, state_po.n_oe]), reg.coef_)
    

    n_parts = len(break_parts)
    n_size = n_trials // n_parts
    assert n_trials % n_parts == 0

    # Shape: trial, n_ko, pattern
    this_states_break = np.zeros((n_trials, n+1, n), dtype=int)
    this_states_build = np.zeros((n_trials, n+1, n), dtype=int)

    this_feat_break = np.zeros((n_trials, n+1, 3), dtype=int)
    this_feat_build = np.zeros((n_trials, n+1, 3), dtype=int)

    for i, break_fname in enumerate(break_parts):

        this_idx = int(break_fname.split('_')[-1].split('.')[0])  

        build_fname = 'build_p_{:02d}_q_{:02d}_idx_{:03d}.npy'.format(p, q, this_idx)

        # Shape: (n_trial, n_ko, pattern)
        break_arr = np.load(break_fname)
        build_arr = np.load(build_fname)

        this_states_break[i*n_size:(i+1)*n_size] = break_arr
        this_states_build[i*n_size:(i+1)*n_size] = build_arr


    for i in range(n_trials):
        for i_ko in range(n+1):
            this_feat_break[i, i_ko] = get_feat(this_states_break[i, i_ko], state_po.adj_mat, state_po.ext_count)
            this_feat_build[i, i_ko] = get_feat(this_states_build[i, i_ko], state_po.adj_mat, state_po.ext_count)

    this_energies_break = np.dot(this_feat_break, reg.coef_)
    this_energies_build = np.dot(this_feat_build, reg.coef_)

    all_states_break['p_{:02d}_q_{:02d}'.format(p,q)] = this_states_break
    all_states_build['p_{:02d}_q_{:02d}'.format(p,q)] = this_states_build

    all_f_break['p_{:02d}_q_{:02d}'.format(p,q)] = this_energies_break
    all_f_build['p_{:02d}_q_{:02d}'.format(p,q)] = this_energies_build


np.savez_compressed('greedy_design_trials.npz', all_states_break=all_states_break, all_states_build=all_states_build,
                                                all_f_build=all_f_build, all_f_break=all_f_break)

