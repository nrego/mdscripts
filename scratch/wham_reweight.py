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

def get_negloghist(data, bins, logweights, do_norm=True):

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
    if do_norm:
        
        norm = np.trapz(np.exp(-negloghist), bins[:-1])
        return negloghist + np.log(norm)

    else:
        return negloghist


# Get PvN, Nvphi, and chi v phi for a set of datapoints and their weights
def extract_and_reweight_data(logweights, ntwid, data, bins, beta_phi_vals, do_norm=True):
    logweights -= logweights.max()
    weights = np.exp(logweights) 
    weights /= weights.sum()

    #pdist, bb = np.histogram(ntwid, bins=bins, weights=weights)
    #pdist_N, bb = np.histogram(data, bins=bins, weights=weights)

    #neglogpdist = -np.log(pdist)
    neglogpdist = get_negloghist(ntwid, bins, logweights, do_norm)

    #neglogpdist -= neglogpdist.min()

    #neglogpdist_N = -np.log(pdist_N)
    neglogpdist_N = get_negloghist(data, bins, logweights, do_norm)

    #neglogpdist_N -= neglogpdist_N.min()
    #embed()
    avg_ntwid = np.zeros_like(beta_phi_vals)
    var_ntwid = np.zeros_like(avg_ntwid)

    avg_data = np.zeros_like(avg_ntwid)
    var_data = np.zeros_like(avg_ntwid)

    ntwid_sq = ntwid**2
    data_sq = data**2

    for i, beta_phi_val in enumerate(beta_phi_vals):
        bias_logweights = logweights - beta_phi_val*ntwid
        bias_logweights -= bias_logweights.max()

        bias_weights = np.exp(bias_logweights)
        bias_weights /= bias_weights.sum()

        this_avg_ntwid = np.dot(bias_weights, ntwid)
        this_avg_ntwid_sq = np.dot(bias_weights, ntwid_sq)
        this_var_ntwid = this_avg_ntwid_sq - this_avg_ntwid**2

        this_avg_data = np.dot(bias_weights, data)
        this_avg_data_sq = np.dot(bias_weights, data_sq)
        this_var_data = this_avg_data_sq - this_avg_data**2

        avg_ntwid[i] = this_avg_ntwid
        var_ntwid[i] = this_var_ntwid
        avg_data[i] = this_avg_data
        var_data[i] = this_var_data

    return (neglogpdist, neglogpdist_N, beta_phi_vals, avg_ntwid, var_ntwid, avg_data, var_data)


beta_phi_vals = np.arange(-1.0,5.04,0.04)
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
all_data_N = all_data_ds['data_N']

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


dat = np.load('boot_fn_payload.dat.npy')

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
