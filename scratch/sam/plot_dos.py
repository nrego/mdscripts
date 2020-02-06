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
from sklearn import linear_model
from scipy.special import binom

plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


### PLOT State functions of full model from WL sampling 
###    DOS for p, q, k_o, n_oo, n_oe
###
#########################################

ds = np.load('sam_dos.npz')
reg = np.load('sam_reg_total.npy').item()

vals_pq = ds['vals_pq']
vals_ko = ds['vals_ko']
vals_f = ds['vals_f']

# Shape: (n_pq_vals, n_ko_vals, n_f_vals)
dos_f = ds['dos_f']

# Our 'toggle' temps
betavals = np.linspace(-1, 1, 101)


# Density of states for a given patch size, ko (pq, choose ko)
# Shape: (n_pq_vals, n_ko_vals)
big_omega = np.zeros(dos_f.shape[:-1])

for i_pq, (p,q) in enumerate(vals_pq):
    for i_ko, ko in enumerate(vals_ko):
        big_omega[i_pq, i_ko] = binom(p*q, ko)


# canonical partittion function
#  shape: (n_betavals, n_pq_vals, n_ko_vals)
z = np.zeros((betavals.size, *dos_f.shape[:-1]))
avg_f_canonical = np.zeros_like(z)
var_f_canonical = np.zeros_like(z)

# Grand canonical part function
#   shape: (n_betavals, n_pq_vals)
Z = np.zeros((betavals.size, vals_pq.shape[0]))
avg_f_gc = np.zeros_like(Z)
var_f_gc = np.zeros_like(Z)

for i, bval in enumerate(betavals):
    this_z = np.dot(dos_f, np.exp(-bval*vals_f))
    this_avg_f = np.dot(dos_f, (np.exp(-bval*vals_f)*vals_f)) / this_z
    this_avg_fsq = np.dot(dos_f, (np.exp(-bval*vals_f)*vals_f**2)) / this_z
    
    z[i] = this_z
    avg_f_canonical[i] = this_avg_f
    var_f_canonical[i] = this_avg_fsq - this_avg_f**2

    # Now for grand-canonical

    this_Z = this_z.sum(axis=1)

    # Any averages with a nan should have a d.o.s. of zero. Here, we make sure this is so.
    mask = np.ma.masked_invalid(this_avg_f).mask
    assert np.array_equal(mask, np.ma.masked_invalid(this_avg_fsq).mask)
    assert np.alltrue(~this_z[mask].astype(bool))
    
    tmp_avg_f = this_avg_f.copy()
    tmp_avg_f[mask] = 0
    this_gc_avg_f = (this_z * tmp_avg_f).sum(axis=1) / this_Z

    tmp_avg_fsq = this_avg_fsq.copy()
    tmp_avg_fsq[mask] = 0
    this_gc_avg_fsq = (this_z * tmp_avg_fsq).sum(axis=1) / this_Z

    Z[i] = this_Z
    avg_f_gc[i] = this_gc_avg_f
    var_f_gc[i] = this_gc_avg_fsq - this_gc_avg_f**2


# (canonical) free energies with temp

bA = -np.log(z)

# (gc) free energies
bPhi = -np.log(Z)


## quick test
import itertools

test_idx = 4
test_p, test_q = vals_pq[test_idx]
test_n = test_p*test_q

test_indices = np.arange(test_n)
test_energies = []

for test_ko in range(test_n+1):
    for test_pt_idx in itertools.combinations(test_indices, test_ko):
        test_pt_idx = np.array(test_pt_idx, dtype=int)
        state = State(test_pt_idx, test_p, test_q)

        test_feat = np.array([test_n, test_p, test_q, state.k_o, state.n_oo, state.n_oe])
        this_e = np.dot(test_feat, reg.coef_)

        test_energies.append(this_e)

test_energies = np.array(test_energies)

assert z[50, test_idx].sum() == Z[50, test_idx] == test_energies.size

