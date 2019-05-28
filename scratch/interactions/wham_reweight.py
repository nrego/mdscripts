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

def get_negloghist(data, bins, logweights):

    max_logweight = logweights.max()
    logweights -= max_logweight
    norm = max_logweight + np.log(np.exp(logweights).sum())
    logweights -= norm

    bin_assign = np.digitize(data, bins) - 1

    negloghist = np.zeros(bins.size-1)

    for i in range(bins.size-1):
        this_bin_mask = bin_assign == i
        this_logweights = logweights[this_bin_mask]
        this_data = data[this_bin_mask]
        if this_data.size == 0:
            continue
        this_logweights_max = this_logweights.max()
        this_logweights -= this_logweights.max()

        this_weights = np.exp(this_logweights)

        negloghist[i] = -np.log(this_weights.sum()) - this_logweights_max

    norm = np.trapz(np.exp(-negloghist), bins[:-1])
    negloghist[negloghist==0] = np.nan
    return negloghist + np.log(norm)



# Get PvN, Nvphi, and chi v phi for a set of datapoints and their weights
def extract_and_reweight_data(logweights, data, bins):
    logweights -= logweights.max()
    weights = np.exp(logweights) 
    weights /= weights.sum()

    neglogpdist = get_negloghist(data, bins, logweights)


    return neglogpdist



mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})

all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data_r = all_data_ds['data']


max_val = np.ceil(all_data_r.max())
db = 0.01 # in nm
bins = np.arange(0, max_val+db, db)


all_neglogpdist = extract_and_reweight_data(all_logweights, all_data_r, bins)


dat = np.load('boot_fn_payload.dat.npy')

n_iter = len(dat)
neglog_pdist = np.zeros((n_iter, bins.size-1))

for i, (this_logweights, boot_data, boot_data_N) in enumerate(dat):
    this_neglogpdist = extract_and_reweight_data(this_logweights, boot_data, bins)

    neglog_pdist[i] = this_neglogpdist


masked_dg = np.ma.masked_invalid(neglog_pdist)

always_null = masked_dg.mask.sum(axis=0) == masked_dg.shape[0]

avg_dg = masked_dg.mean(axis=0)
#avg_dg[always_null] = np.inf
err_dg = masked_dg.std(axis=0, ddof=1)

dat = np.vstack((bins[:-1], all_neglogpdist, err_dg)).T
np.savetxt('PMF.dat', dat, header='r(nm)   beta F(r)  err(beta F(r))   ')

plot_errorbar(bins, all_neglogpdist, err_dg)
plt.show()


