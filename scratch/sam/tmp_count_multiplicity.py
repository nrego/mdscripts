from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob, sys

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import sympy

import itertools
from scipy.special import binom

from scratch.sam.util import *

from functools import reduce

class GetDelta:
    def __init__(self, adj_mat, ext_count, alpha_n_cc, alpha_n_ce):
        self.adj_mat = adj_mat
        self.ext_count = ext_count
        self.alpha_n_cc = alpha_n_cc
        self.alpha_n_ce = alpha_n_ce


    def __call__(self, x0, x1):
        delta_n_cc = 0.5*(np.linalg.multi_dot((x1, self.adj_mat, x1)) - np.linalg.multi_dot((x0, self.adj_mat, x0)))
        delta_n_ce = np.dot(x1-x0, self.ext_count)

        return self.alpha_n_cc * delta_n_cc + self.alpha_n_ce * delta_n_ce


parser = argparse.ArgumentParser('Exhaustively run greedy algorithm')
parser.add_argument('-p', default=4, type=int,
                    help='p (default: %(default)s)')
parser.add_argument('-q', default=4, type=int,
                    help='q (default: %(default)s)')
parser.add_argument('--build-phob', action='store_true', 
                    help='If true, go philic => phobic')


args = parser.parse_args()


homedir = os.environ['HOME']
reg = np.load('sam_reg_coef.npy').item()

alpha1, alpha2, alpha3 = reg.coef_

alpha_n_cc = alpha2
alpha_n_ce = alpha2 - alpha3

p = args.p
q = args.q
n = p*q

pt_idx = np.arange(n, dtype=int) if not args.build_phob else np.array([], dtype=int)

mode = 'build_phil' if not args.build_phob else 'build_phob'

state0 = State(pt_idx, ny=p, nz=q, mode=mode)

x0 = state0.methyl_mask.astype(int)
adj_mat = state0.adj_mat
ext_count = state0.ext_count

delta = GetDelta(adj_mat, ext_count, alpha_n_cc, alpha_n_ce)

# x is methyl mask
# *Breaking* phobicity, so we choose the ko addition
#   that causes the greatest increase in f
def _enumerate_states_break(x, delta, i_round, all_states):

    all_states[i_round].append(x)

    if x.sum() == 0:
        return

    # Go thru candidate hydroxyl placements, select the one(s) with the lowest energy
    avail_indices = np.arange(x.size)[x==1]
    trial_energies = np.zeros_like(avail_indices).astype(float)
    x_cand = np.zeros_like(avail_indices).astype(object)

    for i, cand_idx in enumerate(avail_indices):
        x_trial = x.copy()
        x_cand[i] = x_trial
        x_trial[cand_idx] = 0
        trial_energies[i] = delta(x, x_trial)

    trial_energies = np.round(trial_energies, 5)
    traj_mask = trial_energies == trial_energies.max()

    for new_x in x_cand[traj_mask]:
        _enumerate_states_break(new_x, delta, i_round+1, all_states)

# Breaking phobicity by adding hydroxyls
def enumerate_states(x, delta, mode='build_phil'):
    all_states = dict()
    for i in range(x.size+1):
        all_states[i] = []

    if mode == 'build_phil':
        fn = _enumerate_states_break
    elif mode == 'build_phob':
        fn = _enumerate_states_build

    fn(x, delta, 0, all_states)

    return all_states

print("ENUMERATING STATES FOR MODE {} P: {} Q: {}, index: {}\n".format(mode, p, q))
print("...enumerating states...")
all_states = enumerate_states(x0, delta, mode=mode)
print("...Done\n")

np.save('state_count_p_{:02g}_q_{:02g}_{}'.format(p, q, mode), all_states)

