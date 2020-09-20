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

bins = np.arange(0, max_val+1, 1)


all_neglogpdist, all_neglogpdist_N, beta_phi_vals, avg_ntwid, var_ntwid, avg_N, var_N = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)


logweights = all_logweights
logweights -= all_logweights.max()
weights = np.exp(logweights) 
weights /= weights.sum()
exp_phi = np.exp(- beta*1000*all_data)
maxval = exp_phi.max()

avg_exp_phi = -np.log(np.dot(weights, exp_phi))


dat = np.load('boot_fn_payload.dat.npy', encoding='bytes')

n_iter = len(dat)
neglog_pdist = np.zeros((n_iter, bins.size-1))
neglog_pdist_N = np.zeros((n_iter, bins.size-1))

boot_avg_n = np.zeros((n_iter, beta_phi_vals.size))
boot_var_n = np.zeros((n_iter, beta_phi_vals.size))

for i, (this_logweights, boot_data, boot_data_N) in enumerate(dat):
    this_neglogpdist, this_neglogpdist_N, bphi, this_avg_ntwid, this_var_ntwid, this_avg_n, this_var_n = extract_and_reweight_data(this_logweights, boot_data, boot_data_N, bins, beta_phi_vals, do_norm=False)

    neglog_pdist[i] = this_neglogpdist
    this_neglogpdist_N[this_neglogpdist_N == 0] = np.nan
    neglog_pdist_N[i] = this_neglogpdist_N

    boot_avg_n[i] = this_avg_ntwid
    boot_var_n[i] = this_var_ntwid


err_avg_n = boot_avg_n.std(axis=0, ddof=1)
err_var_n = boot_var_n.std(axis=0, ddof=1)
dat = np.vstack((beta_phi_vals, avg_ntwid, var_ntwid, err_avg_n, err_var_n)).T
np.savetxt('NvPhi.dat', dat, header='beta_phi  <N>  <d N^2>  err(<N>)  err(<dN^2>)')

masked_dg = np.ma.masked_invalid(neglog_pdist)
masked_dg_N = np.ma.masked_invalid(neglog_pdist_N)

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
