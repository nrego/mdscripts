from __future__ import division, print_function

import numpy as np

import argparse
import logging


import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt


## Construct P_v(N) from wham results (after running whamerr.py with '--boot-fn utility_functions.get_weighted_data')


def plot_errorbar(bb, dat, err):
    plt.plot(bb[:-1], dat)
    plt.fill_between(bb[:-1], dat-err, dat+err, alpha=0.5)

temp = 300
k = 8.3144598e-3
beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})


dat = np.load('boot_fn_payload.dat.npy')

n_iter = len(dat)

max_val = 0
for (this_logweights, boot_data, boot_data_N) in dat:
    if boot_data.max() > max_val:
        max_val = np.ceil(boot_data.max())

bins = np.arange(0, max_val+1, 1)
neglog_pdist = np.zeros((n_iter, bins.size-1))
neglog_pdist_N = np.zeros((n_iter, bins.size-1))

for i, (this_logweights, boot_data, boot_data_N) in enumerate(dat):
    weights = np.exp(this_logweights)
    pdist, bb = np.histogram(boot_data, bins=bins, weights=weights)
    pdist_N, bb = np.histogram(boot_data_N, bins=bins, weights=weights)

    this_neglogpdist = -np.log(pdist)
    this_neglogpdist_N = -np.log(pdist_N)

    this_neglogpdist -= this_neglogpdist.min()
    this_neglogpdist_N -= this_neglogpdist_N.min()

    neglog_pdist[i] = this_neglogpdist
    neglog_pdist_N[i] = this_neglogpdist_N

masked_dg = np.ma.masked_invalid(neglog_pdist)
masked_dg_N = np.ma.masked_invalid(neglog_pdist_N)

avg_dg = masked_dg.mean(axis=0)
err_dg = masked_dg.std(axis=0, ddof=1)

avg_dg_N = masked_dg_N.mean(axis=0)
err_dg_N = masked_dg_N.std(axis=0, ddof=1)

plot_errorbar(bb, avg_dg_N, err_dg_N)

plt.show()

