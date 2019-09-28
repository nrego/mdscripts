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
import math


from whamutils import get_negloghist, extract_and_reweight_data

## Construct -ln P_v(N) from wham results (after running whamerr.py with '--boot-fn utility_functions.get_weighted_data')
## also get <N> v phi, and suscept

def plot_errorbar(bb, dat, err):
    plt.plot(bb, dat)
    plt.fill_between(bb, dat-err, dat+err, alpha=0.5)


print('Constructing Nv v phi, chi v phi...')
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
all_data_N = all_data_ds['data_aux']

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)
bins = np.arange(0, max_val+1, 1)

## In kT!
beta_phi_vals = np.linspace(0,4,101)



## Get PvN, <Nv>, chi_v from all data
all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_N, all_chi_N, _ = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)


## Extract errorbars from bootstrap samples
boot_avg_phi = np.zeros((n_iter,beta_phi_vals.size))
boot_chi_phi = np.zeros((n_iter,beta_phi_vals.size))
boot_neglogpdist = np.zeros((n_iter, bins.size-1))
boot_neglogpdist_N = np.zeros((n_iter, bins.size-1))

boot_peak_sus = np.zeros(n_iter)
for i, (this_logweights, boot_data, boot_data_N) in enumerate(dat):

    this_neglogpdist, this_neglogpdist_N, this_avg, this_chi, this_avg_N, this_chi_N, _ = extract_and_reweight_data(this_logweights, boot_data, boot_data_N, bins, beta_phi_vals)

    boot_neglogpdist[i] = this_neglogpdist
    boot_neglogpdist_N[i] = this_neglogpdist_N
    boot_avg_phi[i] = this_avg
    boot_chi_phi[i] = this_chi

    this_peak_sus = beta_phi_vals[np.argmax(this_chi)]
    boot_peak_sus[i] = this_peak_sus


masked_dg = np.ma.masked_invalid(boot_neglogpdist)
masked_dg_N = np.ma.masked_invalid(boot_neglogpdist_N)

always_null = masked_dg.mask.sum(axis=0) == masked_dg.shape[0]
always_null_N = masked_dg_N.mask.sum(axis=0) == masked_dg.shape[0]

avg_dg = masked_dg.mean(axis=0)
err_dg = masked_dg.std(axis=0, ddof=1)

avg_dg_N = masked_dg_N.mean(axis=0)
err_dg_N = masked_dg_N.std(axis=0, ddof=1)

dat = np.vstack((bins[:-1], all_neglogpdist, err_dg)).T
np.savetxt('PvN.dat', dat, header='bins   beta F_v(N)  err(beta F_v(N))   ')

tot_N = boot_avg_phi.mean(axis=0)
err_tot_N = boot_avg_phi.std(axis=0, ddof=1)

tot_chi = boot_chi_phi.mean(axis=0)
err_tot_chi = boot_chi_phi.std(axis=0, ddof=1)

avg_peak_sus = boot_peak_sus.mean()
err_peak_sus = boot_peak_sus.std(ddof=1)

dat = np.vstack((beta_phi_vals, all_avg, err_tot_N, all_chi, err_tot_chi)).T
np.savetxt('NvPhi.dat', dat, header='beta*phi   <N_v>  err(<N_v>)  chi_v   err(chi_v) ')

np.savetxt('peak_sus.dat', np.array([avg_peak_sus, err_peak_sus]), header='beta*phi^*  err(beta phi*)')


print('...Done.')

### Now input all <n_i>_\phi's for a given i ###
print('')
print('Extracting all n_i\'s...')


n_heavies = None


n_i_dat_fnames = sorted(glob.glob('*/rho_data_dump_rad_6.0.dat.npz'))

# Ntwid vals, only taken every 1 ps from each window
all_data_reduced = np.array([])
all_logweights_reduced = np.array([])
# Will have shape (n_heavies, n_tot)
all_data_n_i = None
n_files = len(n_i_dat_fnames)


start_idx = 0
# number of points in each window
shift = all_data.shape[0] // n_files
assert all_data.shape[0] % n_files == 0

## Gather n_i data from each umbrella window (phi value)
for i in range(n_files):
    ## Need to grab every 10th data point (N_v, and weight)
    this_slice = slice(start_idx, start_idx+shift, 10)
    start_idx += shift
    new_data_subslice = all_data[this_slice]
    new_weight_subslice = all_logweights[this_slice]

    all_data_reduced = np.append(all_data_reduced, new_data_subslice)
    all_logweights_reduced = np.append(all_logweights_reduced, new_weight_subslice)
    
    fname = n_i_dat_fnames[i]

    ## Shape: (n_heavies, n_frames) ##
    n_i = np.load(fname)['rho_water'].T
    if n_heavies is None:
        n_heavies = n_i.shape[0]
    else:
        assert n_i.shape[0] == n_heavies

    if all_data_n_i is None:
        all_data_n_i = n_i.copy()
    else:
        all_data_n_i = np.append(all_data_n_i, n_i, axis=1)

print('...Done.')
print('')

print('WHAMing each n_i...')

# Shape: (n_atoms, beta_phi_vals.shape)
avg_nis = np.zeros((n_heavies, beta_phi_vals.size))
chi_nis = np.zeros((n_heavies, beta_phi_vals.size))
cov_nis = np.zeros((n_heavies, beta_phi_vals.size))
for i_atm in range(n_heavies):

    if i_atm % 100 == 0:
        print('  i: {}'.format(i_atm))
        
    neglogpdist, neglogpdist_ni, avg, chi, avg_ni, chi_ni, cov_ni = extract_and_reweight_data(all_logweights_reduced, all_data_reduced, all_data_n_i[i_atm], bins, beta_phi_vals)

    avg_nis[i_atm, :] = avg_ni
    chi_nis[i_atm, :] = chi_ni
    cov_nis[i_atm, :] = cov_ni

np.savez_compressed('ni_rad_weighted.dat', avg=avg_nis, var=chi_nis, cov=cov_nis, beta_phi=phi_vals)


