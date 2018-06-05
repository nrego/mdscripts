from __future__ import division, print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr
import scipy.integrate
from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
import pymbar
import time

import os, glob

from fasthist import normhistnd

from mdtools import ParallelTool
from whamutils import gen_U_nm, kappa, grad_kappa, gen_pdist, gen_data_logweights
import matplotlib as mpl
import matplotlib.pyplot as plt


temp = 300
k = 8.3144598e-3
beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
#mpl.rcParams.update({'titlesize': 42})

fnames = sorted( glob.glob('*/phiout.dat') )

all_dat = np.array([], dtype=dtype)
all_dat_N = np.array([], dtype=dtype)
n_windows = len(fnames)
f_k = np.loadtxt('f_k.dat')
autocorr_time = np.loadtxt('autocorr.dat')

assert len(f_k) == n_windows
n_samples = []

print("loading files...")

ts = None

for fname in fnames:
    ds = dr.loadPhi(fname)
    if ts is None:
        ts = ds.ts
    else:
        assert ts == ds.ts
    dat = np.array(ds.data[500:])

    all_dat = np.append(all_dat, dat[:,1])
    all_dat_N = np.append(all_dat_N, dat[:,0])
    n_samples.append(dat.shape[0])

autocorr_block = np.ceil(1+2*autocorr_time/ts).astype(int)
n_samples = np.array(n_samples).astype(float)
uncorr_n_samples = n_samples / autocorr_block

n_tot = all_dat.size

assert n_samples.sum() == n_tot
assert n_samples.size == n_windows

print("    ...done")

bias_mat = np.zeros((n_tot, n_windows), dtype=dtype)

# Fill up bias matrix
print("filling up bias matrix...")
nstars = []
avg_by_nstar = []
for i, (fname, ds) in enumerate(dr.datasets.iteritems()):
    assert fnames[i] == fname
    nstars.append(ds.Nstar)
    bias_mat[:,i] = beta*((ds.phi*all_dat) + (ds.kappa/2.0)*(all_dat-ds.Nstar)**2)
    avg_by_nstar.append(ds.data[500:]['$\~N$'].mean())

print("    ...done")


#Q = -bias_mat + f_k + np.log(uncorr_n_samples)
#max_vals = Q.max(axis=1)
#Q -= max_vals[:,None]

#logweights = -( np.log( np.exp(Q).sum(axis=1) ) + max_vals )

logweights = gen_data_logweights(bias_mat, f_k, uncorr_n_samples)
weights = np.exp(logweights)
#weights /= weights.sum()

phi_vals = np.arange(0, 10.1, 0.1)

avg_ns = []
var_ns = []

# x1 = <N>_\phi / <N>_0
avg_x1s = []
var_x1s = []

avg_x2s = []
var_x2s = []

avg_n0 = 0
var_n0 = 0
for phi in phi_vals:
    bias = beta*phi*all_dat
    this_logweights = logweights - bias
    this_logweights -= this_logweights.max()
    this_weights = np.exp(this_logweights)
    this_weights /= this_weights.sum()

    this_avg_n = np.dot(this_weights, all_dat)
    this_avg_n_sq = np.dot(this_weights, all_dat**2)
    this_var = this_avg_n_sq - this_avg_n**2

    avg_ns.append(this_avg_n)
    var_ns.append(this_var)

    if phi == 0:
        avg_n0 = this_avg_n
        var_n0 = this_var

    alpha = 1
    n0 = avg_n0

    this_avg_x1 = (alpha/n0) * this_avg_n
    this_var_x1 = (alpha/n0)**2 * this_var
    avg_x1s.append(this_avg_x1)
    var_x1s.append(this_var_x1)

    alpha = (avg_n0/np.sqrt(var_n0))

    this_avg_x2 = (alpha/n0) * this_avg_n
    this_var_x2 = (alpha/n0)**2 * this_var
    avg_x2s.append(this_avg_x2)
    var_x2s.append(this_var_x2)

dat_arr = np.dstack((phi_vals, avg_ns, var_ns)).squeeze()
np.savetxt('n_v_phi.dat', dat_arr)

dat_arr1 = np.dstack((phi_vals, avg_x1s, var_x1s)).squeeze()
np.savetxt('x1_v_phi.dat', dat_arr)

dat_arr = np.dstack((phi_vals, avg_x2s, var_x2s)).squeeze()
np.savetxt('x2_v_phi.dat', dat_arr)
