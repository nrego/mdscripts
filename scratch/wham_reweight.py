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
from whamutils import gen_U_nm, kappa, grad_kappa, gen_pdist
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
logweights = np.loadtxt('logweights.dat')

assert len(logweights) == n_windows
n_samples = []

print("loading files...")

for fname in fnames:
    ds = dr.loadPhi(fname)
    dat = np.array(ds.data[500:])

    all_dat = np.append(all_dat, dat[:,1])
    all_dat_N = np.append(all_dat_N, dat[:,0])
    n_samples.append(dat.shape[0])


n_samples = np.array(n_samples).astype(float)
n_tot = all_dat.size

assert n_samples.sum() == n_tot
assert n_samples.size == n_windows

print("    ...done")

bias_mat = np.zeros((n_tot, n_windows), dtype=dtype)

# Fill up bias matrix
print("filling up bias matrix...")

for i, (fname, ds) in enumerate(dr.datasets.iteritems()):
    assert fnames[i] == fname
    bias_mat[:,i] = beta*((ds.phi*all_dat) + (ds.kappa/2.0)*(all_dat-ds.Nstar)**2)

print("    ...done")

Q = bias_mat - logweights
max_val = Q.max()
Q -= max_val



