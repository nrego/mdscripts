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


def plot_errorbar(bb, dat, err):
    plt.plot(bb[:-1], dat)
    plt.fill_between(bb[:-1], dat-err, dat+err, alpha=0.5)

# Get PvN, Nvphi, and chi v phi for a set of datapoints and their weights
def extract_and_reweight_data(logweights, ntwid, data, bins):
    logweights -= logweights.max()
    weights = np.exp(logweights) 

    pdist, bb = np.histogram(ntwid, bins=bins, weights=weights)
    pdist_N, bb = np.histogram(data, bins=bins, weights=weights)

    neglogpdist = -np.log(pdist)
    neglogpdist -= neglogpdist.min()

    neglogpdist_N = -np.log(pdist_N)
    neglogpdist_N -= neglogpdist_N.min()


    return (neglogpdist, neglogpdist_N)



temp = 300
k = 8.3144598e-3
beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})

all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_N']

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)

bins = np.arange(0, max_val+1, 1)


all_neglogpdist, all_neglogpdist_N = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins)

dat = np.load('boot_fn_payload.dat.npy')

n_iter = len(dat)
neglog_pdist = np.zeros((n_iter, bins.size-1))
neglog_pdist_N = np.zeros((n_iter, bins.size-1))

for i, (this_logweights, boot_data, boot_data_N) in enumerate(dat):
    this_neglogpdist, this_neglogpdist_N = extract_and_reweight_data(this_logweights, boot_data, boot_data_N, bins)

    neglog_pdist[i] = this_neglogpdist
    neglog_pdist_N[i] = this_neglogpdist_N

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
