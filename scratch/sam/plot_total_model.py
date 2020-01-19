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

plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})




ds_06_06 = np.load('sam_pattern_06_06.npz')
ds_04_04 = np.load('sam_pattern_04_04.npz')
ds_04_09 = np.load('sam_pattern_04_09.npz')


energies_06_06 = ds_06_06['energies']
energies_04_04 = ds_04_04['energies']
energies_04_09 = ds_04_09['energies']

err_06_06 = ds_06_06['err_energies']
err_04_04 = ds_04_04['err_energies']
err_04_09 = ds_04_09['err_energies']
err_06_06 = np.ones_like(err_06_06)
err_04_04 = np.ones_like(err_04_04)
err_04_09 = np.ones_like(err_04_09)

states_06_06 = ds_06_06['states']
states_04_04 = ds_04_04['states']
states_04_09 = ds_04_09['states']

feat_06_06 = extract_from_states(states_06_06)
feat_04_04 = extract_from_states(states_04_04)
feat_04_09 = extract_from_states(states_04_09)