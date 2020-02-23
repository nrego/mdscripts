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
reg = np.load('sam_reg_pooled.npy').item()

alpha1, alpha2, alpha3 = reg.coef_

p = args.p
q = args.q
n = p*q

pt_idx = np.arange(n, dtype=int) if not args.build_phob else np.array([], dtype=int)

mode = 'build_phil' if not args.build_phob else 'build_phob'
state0 = State(pt_idx, ny=p, nz=q, mode=mode)

def enumerate_states(state):

    n_reachable = state.avail_indices.size

    if n_reachable == 0:
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
        if n_reachable == state.N:
            print("doing {} of {}".format(i+1, state.children.size))
            sys.stdout.flush()
            
        enumerate_states(child)


def print_states(state):
    plt.close('all')
    state.plot()
    n_avail = state.avail_indices.size
    plt.savefig('{}/Desktop/state_{}'.format(homedir, n_avail))

    if len(state.children) == 0:
        return

    print_states(state.children[0])

# Go thru state trajectory tree, enumerate patterns
#   at each k_o

def _make_prob_dist(state, state_count):
    idx = state.k_o

    state_count[idx].append(state.methyl_mask)

    for child in state.children:
        _make_prob_dist(child, state_count)

def make_prob_dist(state):
    state_count = [ [] for i in range(state.N+1) ]

    _make_prob_dist(state, state_count)

    state_count = [np.array(arr) for arr in state_count]

    return state_count  

print("ENUMERATING STATES FOR MODE {} P: {} Q: {}\n".format(mode, p, q))
enumerate_states(state0)
print("...Done\n")
print("...enumerating states...")
state_count = make_prob_dist(state0)
del state0
print("...Done...Saving...\n")
np.save('state_count_p_{:02g}_q_{:02g}_{}'.format(p, q, mode), state_count)
print("...Done!")
