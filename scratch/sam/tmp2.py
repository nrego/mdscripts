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



parser = argparse.ArgumentParser('exhaustively run greedy algorithm')
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

p = args.p
q = args.q
n = p*q

pt_idx = np.arange(n, dtype=int) if not args.build_phob else np.array([], dtype=int)

mode = 'build_phil' if not args.build_phob else 'build_phob'
state0 = State(pt_idx, ny=p, nz=q, mode=mode)

def enumerate_states(state, break_out=0):

    n_reachable = state.avail_indices.size

    if n_reachable == break_out:
        return 

    trial_states = np.empty(n_reachable, dtype=object)
    trial_energies = np.zeros(n_reachable)

    for i, newstate in enumerate(state.gen_next_pattern()):

        trial_states[i] = newstate
        trial_energies[i] = alpha2*(newstate.n_oo) + alpha3*(newstate.n_oe)

    if state.mode == 'build_phil':
        mask = np.isclose(trial_energies, trial_energies.max())
    else:
        mask = np.isclose(trial_energies, trial_energies.min())

    state.children = trial_states[mask]
    
    for i, child in enumerate(state.children):
        if n_reachable >= state.N - 2:
            print("doing {} of {}".format(i+1, state.children.size))
            sys.stdout.flush()

        enumerate_states(child, break_out=break_out)

import time
def print_states(state):
    np.random.seed(int(time.time()))

    plt.close('all')
    state.plot()
    n_avail = state0.N - state.avail_indices.size
    plt.savefig('{}/Desktop/state_{}'.format(homedir, n_avail))

    n_children = len(state.children)
    if n_children == 0:
        return

    idx = np.random.randint(n_children)
    print_states(state.children[idx])

# Go thru state trajectory tree, enumerate patterns
#   at each k_o

def _make_prob_dist(state, state_count, energies, reg):
    idx = state.k_o
    ener = np.dot(np.array([state.k_o, state.n_oo, state.n_oe]), reg.coef_)
    state_count[idx].append(state.methyl_mask)
    energies[idx].append(ener)

    for child in state.children:
        _make_prob_dist(child, state_count, energies, reg)

def make_prob_dist(state, reg):
    state_count = [ [] for i in range(state.N+1) ]
    energies = [ [] for i in range(state.N+1) ]

    _make_prob_dist(state, state_count, energies, reg)

    state_count = [np.array(arr) for arr in state_count]
    energies = [np.array(arr) for arr in energies]

    return state_count, energies 

## Finding initial states
print("ENUMERATING STATES FOR MODE {} P: {} Q: {}\n".format(mode, p, q))
print("(enumerating intitial trial moves)")
enumerate_states(state0, break_out=state0.N-2)

all_states = []

for c1 in state0.children:
    for c2 in c1.children:
        all_states.append(c2)

print("number of states: {}".format(len(all_states)))



idx = 0
for c1 in state0.children:
    for c2 in c1.children:
        expt_state = all_states[idx]
        assert np.array_equal(c2.pt_idx, expt_state.pt_idx)

        mystr = 'state_p_{:02g}_q_{:02g}_idx_{:03g}_{}.npy'.format(p, q, idx, mode)
        print("loading {}".format(mystr))
        newstate = np.load(mystr).item()
        assert np.array_equal(c2.pt_idx, newstate.pt_idx)

        c2.children = newstate.children
        idx += 1

print("counting up all states...")
state_count, energies = make_prob_dist(state0, reg)
np.save('state_p_{:02g}_q_{:02g}_{}.npy'.format(p, q, mode), state0)

ko = np.arange(state0.N+1)
avg_energies = np.array([e.mean() for e in energies])
var_energies = np.array([e.var() for e in energies])

np.savez_compressed('state_count_p_{:02g}_q_{:02g}_{}'.format(p, q, mode), state_count=state_count, energies=energies)


