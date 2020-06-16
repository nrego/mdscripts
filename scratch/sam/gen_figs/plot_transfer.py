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

import itertools

from sklearn import datasets, linear_model
from sklearn.cluster import AgglomerativeClustering


np.set_printoptions(precision=3)



p = 6
q = 6

# For SAM schematic pattern plotting
figsize = (10,10)

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':34})

homedir = os.environ['HOME']

energies_06_06, ols_feat_vec_06_06, states_06_06 = extract_from_ds('data/sam_pattern_06_06.npz')
err_energies_06_06 = np.load('data/sam_pattern_{:02d}_{:02d}.npz'.format(p,q))['err_energies']
weights_06_06 = 1 / err_energies_06_06**2

perf_mse_06_06, perf_wt_mse_06_06, perf_r2_06_06, err_06_06, reg_06_06 = fit_multi_k_fold(ols_feat_vec_06_06, energies_06_06, weights=weights_06_06, do_weighted=False)

state_06_06 = states_06_06[np.argwhere(ols_feat_vec_06_06[:,0] == 0).item()]
