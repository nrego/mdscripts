
import numpy as np

import argparse
import logging

import scipy

import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed
import math

import sys
import pymbar
from whamutils import get_negloghist, extract_and_reweight_data

## Construct -ln P_v(N) from wham results (after running whamerr.py with '--boot-fn utility_functions.get_weighted_data')
## also get <N> v phi, and suscept

def plot_errorbar(bb, dat, err):
    plt.plot(bb, dat)
    plt.fill_between(bb, dat-err, dat+err, alpha=0.5)

def subsample(dat, n_samples, uncorr_n_samples):
    cum_n_samples = np.append(0, np.cumsum(n_samples))
    boot_dat = np.zeros(uncorr_n_samples.sum())
    np.random.seed()

    uncorr_start_idx = 0
    boot_indices = np.array([], dtype=int)
    for i, uncorr_n_sample in enumerate(uncorr_n_samples):
        start = cum_n_samples[i]
        end = cum_n_samples[i+1]
        # Indices from dat for this window
        avail_indices = np.arange(start, end, dtype=int)
        this_indices = np.random.choice(avail_indices, size=uncorr_n_sample, replace=True)

        boot_dat[uncorr_start_idx:uncorr_start_idx+uncorr_n_sample] = dat[this_indices]

        uncorr_start_idx += uncorr_n_sample
        boot_indices = np.append(boot_indices, this_indices)

    return boot_dat, boot_indices

print('Constructing Nv v phi, chi v phi...')
sys.stdout.flush()
temp = 300

beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})


### EXTRACT DATA ###

all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_aux']

boot_indices = np.load('boot_indices.dat.npy')
dat = np.load('boot_fn_payload.dat.npy')
n_iter = boot_indices.shape[0]

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)
bins = np.arange(0, max_val+1, 1)

## In kT!
beta_phi_vals = np.linspace(0,4,101)


## Get PvN, <Nv>, chi_v from all data ###
all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_N, all_chi_N, _ = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)

### Now input all <n_i>_\phi's for a given i ###
print('')
print('Extracting all n_i\'s...')
sys.stdout.flush()


n_i_dat_fnames = np.append(sorted(glob.glob('phi*/rho_data_dump_rad_6.0.dat.npz')), sorted(glob.glob('nstar*/rho_data_dump_rad_6.0.dat.npz')))
all_data_n_i = None

n_heavies = np.load(n_i_dat_fnames[0])['rho_water'].shape[1]
taus = np.zeros((len(n_i_dat_fnames), n_heavies))
taus[:] = np.inf
n_samples = np.zeros((len(n_i_dat_fnames), n_heavies))
uncorr_n_samples = np.zeros((len(n_i_dat_fnames), n_heavies))

## Gather n_i data from each umbrella window (phi value)
for i_f, fname in enumerate(n_i_dat_fnames):
    print("loading fname: {}".format(fname), end='\r')
    ## Shape: (n_heavies, n_frames) ##
    n_i = np.load(fname)['rho_water'].T

    for i_atm in range(n_heavies):
        n_samples[i_f, i_atm] = n_i[i_atm].size
        try:
            this_tau = pymbar.timeseries.integratedAutocorrelationTime(n_i[i_atm])
            taus[i_f, i_atm] = this_tau
            uncorr_n_samples[i_f, i_atm] = n_i[i_atm].size // (1 + 2*this_tau)
        except:
            pass

    if n_heavies is None:
        n_heavies = n_i.shape[0]
    else:
        assert n_i.shape[0] == n_heavies

    if all_data_n_i is None:
        all_data_n_i = n_i.copy()
    else:
        all_data_n_i = np.append(all_data_n_i, n_i, axis=1)

uncorr_n_samples = uncorr_n_samples.astype(int)
print('...Done.')
print('')

print('WHAMing each n_i...')
sys.stdout.flush()

# Shape: (n_atoms, beta_phi_vals.shape)
avg_nis = np.zeros((n_heavies, beta_phi_vals.size))
smooth_avg_nis = np.zeros_like(avg_nis)
chi_nis = np.zeros((n_heavies, beta_phi_vals.size))
cov_nis = np.zeros((n_heavies, beta_phi_vals.size))
smooth_cov_nis = np.zeros_like(cov_nis)

for i_atm in range(n_heavies):

    import time
    start = time.time()


    neglogpdist, neglogpdist_ni, avg, chi, avg_ni, chi_ni, cov_ni = extract_and_reweight_data(all_logweights, all_data, all_data_n_i[i_atm], bins, beta_phi_vals)

    boot_avg_ni = np.zeros((n_iter, beta_phi_vals.size))
    for i_boot in range(n_iter):
        #this_boot_indices = boot_indices[i_boot]
        #(this_logweights, boot_data, boot_data_N) = dat[i_boot]
        boot_data, this_boot_indices = subsample(all_data_n_i[i_atm], n_samples[:,i_atm], uncorr_n_samples[:,i_atm])

        #_, _, _, _, this_boot_avg_ni, _, _ = extract_and_reweight_data(all_logweights[boot_indices], boot_data, all_data_n_i[i_atm][boot_indices[i_boot]], bins, beta_phi_vals)
        _, _, _, _, this_boot_avg_ni, _, _ = extract_and_reweight_data(all_logweights[this_boot_indices], all_data[this_boot_indices], boot_data, bins, beta_phi_vals)
        boot_avg_ni[i_boot] = this_boot_avg_ni

    err_avg_ni = np.ma.masked_invalid(boot_avg_ni).std(axis=0, ddof=1)
    spl = scipy.interpolate.UnivariateSpline(beta_phi_vals, avg_ni, w=1/err_avg_ni)

    avg_nis[i_atm, :] = avg_ni
    chi_nis[i_atm, :] = chi_ni
    cov_nis[i_atm, :] = cov_ni

    smooth_avg_nis[i_atm, :] = spl(beta_phi_vals)
    smooth_cov_nis[i_atm, :] = -spl(beta_phi_vals, 1)
    end = time.time()

    if i_atm % 1 == 0:
        print('  i: {}, ({:.2f}s)'.format(i_atm, end-start), end='\r')
        sys.stdout.flush()
        
np.savez_compressed('ni_rad_weighted.dat', avg=avg_nis, var=chi_nis, cov=cov_nis, smooth_avg=smooth_avg_nis, smooth_cov=smooth_cov_nis, beta_phi=beta_phi_vals)

