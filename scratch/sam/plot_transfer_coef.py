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

from scratch.sam.util import *


def extract_from_states(states):
    feat_vec = np.zeros((states.size, 3))

    for i, state in enumerate(states):
        feat_vec[i] = state.k_o, state.n_oo, state.n_oe


    return feat_vec


plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


### PLOT Transferability of coefs for 4x4, 6x6, 4x9 ####
#########################################

ds_06_06 = np.load('sam_pattern_06_06.npz')
ds_04_04 = np.load('sam_pattern_04_04.npz')
ds_04_09 = np.load('sam_pattern_04_09.npz')


energies_06_06 = ds_06_06['energies']
energies_04_04 = ds_04_04['energies']
energies_04_09 = ds_04_09['energies']

states_06_06 = ds_06_06['states']
states_04_04 = ds_04_04['states']
states_04_09 = ds_04_09['states']

feat_06_06 = extract_from_states(states_06_06)
feat_04_04 = extract_from_states(states_04_04)
feat_04_09 = extract_from_states(states_04_09)







