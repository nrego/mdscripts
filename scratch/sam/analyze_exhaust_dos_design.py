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

import shutil

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'lines.linewidth':2})
mpl.rcParams.update({'lines.markersize':8})
mpl.rcParams.update({'legend.fontsize':10})

COLOR_BREAK = '#1f77b4'
COLOR_BUILD = '#7f7f7f'

## Analysze entropy from exhaustive search over greedy trajectory design
homedir = os.environ['HOME']

p = 6
q = 6

n = p*q

indices = np.arange(n)

dummy_state = State(indices, ny=p, nz=q)
adj_mat = dummy_state.adj_mat
ext_count = dummy_state.ext_count

reg = np.load("sam_reg_coef.npy").item()

# ko, noo, noe
alpha1, alpha2, alpha3 = reg.coef_

alpha_kc = -alpha1 - 6*alpha2
alpha_n_cc = alpha2
alpha_n_ce = alpha2 - alpha3

headdir = 'p_{:02d}_q_{:02d}'.format(p,q)

n_accessible_states_break = np.zeros(n+1)
n_accessible_states_build = np.zeros(n+1)

shutil.copy('{}/break_phob/kc_0000.pkl'.format(headdir), '{}/build_phob/'.format(headdir))
shutil.copy('{}/build_phob/kc_{:04d}.pkl'.format(headdir, n), '{}/break_phob/'.format(headdir))


for i in range(n+1):
    kc = i
    with open('{}/break_phob/kc_{:04d}.pkl'.format(headdir, i), 'rb') as fin:
        state_count_break = pickle.load(fin)
    with open('{}/build_phob/kc_{:04d}.pkl'.format(headdir, i), 'rb') as fin:
        state_count_build = pickle.load(fin)

    methyl_masks_break = np.array([np.fromstring(state_byte, bool) for state_byte in state_count_break.keys()])
    methyl_masks_build = np.array([np.fromstring(state_byte, bool) for state_byte in state_count_build.keys()])
    n_state_break = methyl_masks_break.shape[0]
    n_state_build = methyl_masks_build.shape[0]

    x_break = methyl_masks_break.astype(int)
    x_build = methyl_masks_build.astype(int)

    n_cc_break = 0.5*np.linalg.multi_dot((x_break, adj_mat, x_break.T))
    n_cc_build = 0.5*np.linalg.multi_dot((x_build, adj_mat, x_build.T))

    n_ce_break = np.dot(x_break, ext_count)
    n_ce_build = np.dot(x_build, ext_count)

    d_energy_break = alpha_kc * kc + alpha_n_cc * n_cc_break + alpha_n_ce * n_ce_break
    d_energy_build = alpha_kc * kc + alpha_n_cc * n_cc_build + alpha_n_ce * n_ce_build

    #print("k_c: {}. n states: {}".format(i, n_state))

    n_accessible_states_break[i] = n_state_break
    n_accessible_states_build[i] = n_state_build


assert n_accessible_states_build[0] == n_accessible_states_build[-1] == n_accessible_states_break[0] == n_accessible_states_break[-1] == 1

plt.plot(np.arange(n+1), np.log(n_accessible_states_break), '-o', color=COLOR_BREAK)
plt.plot(np.arange(n+1), np.log(n_accessible_states_build), '-o', color=COLOR_BUILD)
plt.show()
