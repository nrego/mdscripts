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

import numpy as np
import sympy

import itertools

from scratch.sam.util import *

import pickle

## Analysze entropy from exhaustive search over greedy trajectory design

p = 6
q = 6

n = p*q



n_accessible_states_break = np.zeros(n+1)
n_accessible_states_build = np.zeros(n+1)

for i in range(n+1):

    with open('break_phob/kc_{:04d}.pkl'.format(i), 'rb') as fin:
        state_count_break = pickle.load(fin)
    with open('build_phob/kc_{:04d}.pkl'.format(i), 'rb') as fin:
        state_count_build = pickle.load(fin)

    methyl_masks = np.array([np.fromstring(state_byte, bool) for state_byte in state_count.keys()])
    n_state = methyl_masks.shape[0]

    print("k_c: {}. n states: {}".format(i, n_state))

    n_accessible_states[i] = n_state

#n_accessible_states[-1] = 1
plt.plot(np.arange(n+1), np.log(n_accessible_states), 'o')


