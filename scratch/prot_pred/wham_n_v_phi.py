from __future__ import division, print_function

import numpy as np

import argparse
import logging


import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed

## Construct -ln P_v(N) from wham results (after running whamerr.py with '--boot-fn utility_functions.get_weighted_data')
## also get <N> v phi, and suscept

def plot_errorbar(bb, dat, err):
    plt.plot(bb[:-1], dat)
    plt.fill_between(bb[:-1], dat-err, dat+err, alpha=0.5)

# Get PvN, Nvphi, and chi v phi for a set of datapoints and their weights
def extract_and_reweight_data(logweights, data, data_N, bins, phi_vals):
    logweights -= logweights.max()
    weights = np.exp(logweights) 

    pdist, bb = np.histogram(data, bins=bins, weights=weights)
    pdist_N, bb = np.histogram(data_N, bins=bins, weights=weights)

    neglogpdist = -np.log(pdist)
    neglogpdist -= neglogpdist.min()

    neglogpdist_N = -np.log(pdist_N)
    neglogpdist_N -= neglogpdist_N.min()

    avg_N = np.zeros_like(phi_vals)
    chi = np.zeros_like(phi_vals)

    # Now get average N and chi for each phi
    for idx_phi, phi_val in enumerate(phi_vals):
        bias = phi_val * data

        bias_logweights = logweights - bias
        bias_logweights -= bias_logweights.max()
        bias_weights = np.exp(bias_logweights)
        bias_weights /= bias_weights.sum()

        this_avg_N = np.dot(data, bias_weights)
        this_avg_N_sq = np.dot(data**2, bias_weights)
        this_chi = this_avg_N_sq - this_avg_N**2

        avg_N[idx_phi] = this_avg_N
        chi[idx_phi] = this_chi

    
    return (neglogpdist, neglogpdist_N, avg_N, chi)

temp = 300

beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})


dat = np.load('boot_fn_payload.dat.npy')

n_iter = len(dat)


all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_N']

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)

## In kT!
phi_vals = np.linspace(0,4,21)
avg_N_phi = np.zeros((n_iter,phi_vals.size))
chi_phi = np.zeros((n_iter,phi_vals.size))

bins = np.arange(0, max_val+1, 1)
neglog_pdist = np.zeros((n_iter, bins.size-1))
neglog_pdist_N = np.zeros((n_iter, bins.size-1))


## Get PvN, <Nv>, chi_v from all data
all_neglogpdist, all_neglogpdist_N, all_avg_N, all_chi = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, phi_vals)


## Extract errorbars from bootstrap samples
peak_sus = np.zeros(n_iter)
for i, (this_logweights, boot_data, boot_data_N) in enumerate(dat):

    boot_neglogpdist, boot_neglogpdist_N, boot_avg_N, boot_chi = extract_and_reweight_data(this_logweights, boot_data, boot_data_N, bins, phi_vals)

    neglog_pdist[i] = boot_neglogpdist
    neglog_pdist_N[i] = boot_neglogpdist_N
    avg_N_phi[i] = boot_avg_N
    chi_phi[i] = boot_chi

    this_peak_sus = phi_vals[np.argmax(boot_chi)]
    peak_sus[i] = this_peak_sus

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

tot_N = avg_N_phi.mean(axis=0)
err_tot_N = avg_N_phi.std(axis=0, ddof=1)

tot_chi = chi_phi.mean(axis=0)
err_tot_chi = chi_phi.std(axis=0, ddof=1)

avg_peak_sus = peak_sus.mean()
err_peak_sus = peak_sus.std(ddof=1)

dat = np.vstack((phi_vals, all_avg_N, err_tot_N, all_chi, err_tot_chi)).T
np.savetxt('Nvphi.dat', dat, header='beta*phi   <N_v>  err(<N_v>)  chi_v   err(chi_v) ')

np.savetxt('peak_sus.dat', np.array([avg_peak_sus, err_peak_sus]), header='beta*phi^*  err(beta phi*)')

