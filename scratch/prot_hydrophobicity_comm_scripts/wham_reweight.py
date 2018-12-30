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
def extract_and_reweight_data(logweights, ntwid, data, bins, phi_vals):
    logweights -= logweights.max()
    weights = np.exp(logweights) 

    pdist, bb = np.histogram(ntwid, bins=bins, weights=weights)
    pdist_N, bb = np.histogram(data, bins=bins, weights=weights)

    neglogpdist = -np.log(pdist)
    neglogpdist -= neglogpdist.min()

    neglogpdist_N = -np.log(pdist_N)
    neglogpdist_N -= neglogpdist_N.min()

    avg_N = np.zeros_like(phi_vals)
    chi = np.zeros_like(phi_vals)

    # Now get average N and chi for each phi
    for idx_phi, phi_val in enumerate(phi_vals):
        bias = phi_val * ntwid

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

phi_vals = np.linspace(0,4,21)
def get_output(dirname, bins):
    dat = np.load('{}/boot_fn_payload.dat.npy'.format(dirname))


    n_iter = len(dat)


    all_data_ds = np.load('{}/all_data.dat.npz'.format(dirname))
    all_logweights = all_data_ds['logweights']
    all_data = all_data_ds['data']
    all_data_N = all_data_ds['data_N']

    max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)

    ## In kT!
    
    avg_N_phi = np.zeros((n_iter,phi_vals.size))
    chi_phi = np.zeros((n_iter,phi_vals.size))

    #bins = np.arange(0, max_val+1, 1)
    neglog_pdist = np.zeros((n_iter, bins.size-1))
    neglog_pdist_N = np.zeros((n_iter, bins.size-1))


    ## Get PvN, <Nv>, chi_v from all data
    return extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, phi_vals)

bins = np.arange(0, 160, 1)
wt_neglogpdist, all_neglogpdist_N, wt_avg_N, wt_chi = get_output('wham_org', bins)
mut_neglogpdist, all_neglogpdist_N, mut_avg_N, mut_chi = get_output('wham_mut', bins)

