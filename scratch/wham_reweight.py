from __future__ import division, print_function

import numpy as np

import argparse
import logging


import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython import embed

from constants import k
from whamutils import get_negloghist, extract_and_reweight_data

## Construct P_v(N) from wham results (after running whamerr.py with '--boot-fn utility_functions.get_weighted_data')


def plot_errorbar(bb, dat, err, **kwargs):
    plt.plot(bb, dat, **kwargs)
    plt.fill_between(bb, dat-err, dat+err, alpha=0.5)


beta_phi_vals = np.arange(0,6.02,0.02)
temp = 300
#k = 8.3144598e-3
beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})

all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_aux']

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)

# Ntwid and N bins
bins = np.arange(0, max_val+1, 1)

all_neglogpdist, all_neglogpdist_N, avg_ntwid, var_ntwid, avg_N, var_N, cov_data = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)


## Now get errorbars from bootstrap samples
dat = np.load('boot_fn_payload.dat.npy', encoding='bytes')

boot_n_iter = len(dat)
boot_neglogpdist = np.zeros((boot_n_iter, bins.size-1))
boot_neglogpdist_N = np.zeros((boot_n_iter, bins.size-1))

boot_avg_N = np.zeros((boot_n_iter, beta_phi_vals.size))
boot_var_N = np.zeros((boot_n_iter, beta_phi_vals.size))

# Extract data for each bootstrap
for i, (this_logweights, this_data, this_data_N) in enumerate(dat):
    this_neglogpdist, this_neglogpdist_N, this_avg_ntwid, this_var_ntwid, this_avg_N, this_var_N, this_cov = extract_and_reweight_data(this_logweights, this_data, this_data_N, bins, beta_phi_vals)

    boot_neglogpdist[i] = this_neglogpdist
    boot_neglogpdist_N[i] = this_neglogpdist_N

    boot_avg_N[i] = this_avg_ntwid
    boot_var_N[i] = this_var_ntwid


err_avg_n = boot_avg_N.std(axis=0, ddof=1)
err_var_n = boot_var_N.std(axis=0, ddof=1)
dat = np.vstack((beta_phi_vals, avg_ntwid, var_ntwid, err_avg_n, err_var_n)).T
np.savetxt('NvPhi.dat', dat, header='beta_phi  <N>  <d N^2>  err(<N>)  err(<dN^2>)')

masked_dg = np.ma.masked_invalid(boot_neglogpdist)
masked_dg_N = np.ma.masked_invalid(boot_neglogpdist_N)

always_null = masked_dg.mask.sum(axis=0) == masked_dg.shape[0]
always_null_N = masked_dg_N.mask.sum(axis=0) == masked_dg.shape[0]

avg_dg = masked_dg.mean(axis=0)
avg_dg[always_null] = np.inf
err_dg = masked_dg.std(axis=0, ddof=1)

avg_dg_N = masked_dg_N.mean(axis=0)
avg_dg_N[always_null_N] = np.inf
err_dg_N = masked_dg_N.std(axis=0, ddof=1)

dat = np.vstack((bins[:-1], all_neglogpdist_N, err_dg_N)).T
np.savetxt('PvN.dat', dat, header='bins   beta F_v(N)  err(beta F_v(N))   ')
