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


reg_coef = np.load('sam_reg_pooled.npy').item()
reg_int = np.load('sam_reg_inter.npy').item()

inter = reg_int.intercept_
# for P, Q, k_o, n_oo, n_oe
coefs = np.concatenate((reg_int.coef_, reg_coef.coef_))

ds = np.load('sam_pattern_pooled.npz')
energies = ds['energies']
states = ds['states']
feat_vec = ds['feat_vec']
errs = 1 / ds['weights']

# Total number of unique edges for each size (6x6, 4x9, 4x4)
n_edges_int = np.array([85., 83., 33.])
n_edges_ext = np.array([46., 50., 30.])

n_edges_tot = n_edges_int + n_edges_ext
pq = np.array([[36, 12],
               [36, 13],
               [16, 8]])
n_ext = np.array([20, 22, 12])

p, q, ko, n_oo, n_oe, kc, n_mm, n_me, n_mo = np.split(feat_vec, indices_or_sections=9, axis=1)

pq = p*q
p_p_q = p+q

# Fit a dummy regression and then change its coeffs
myfeat = np.hstack((pq, p_p_q, ko, n_oo, n_oe))
reg = linear_model.LinearRegression()
reg.fit(myfeat, energies)

reg.intercept_ = reg_int.intercept_
reg.coef_[:2] = reg_int.coef_
reg.coef_[2:] = reg_coef.coef_