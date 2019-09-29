from __future__ import division, print_function

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


### EXTRACT DATA ###
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



## Get PvN, <Nv>, chi_v from all data ###
all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_N, all_chi_N, _ = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)


## Extract errorbars for avg N v phi ###
boot_avg_phi = np.zeros((n_iter,beta_phi_vals.size))
# For reference, but won't be used bc too noisy
boot_chi_phi = np.zeros((n_iter,beta_phi_vals.size))
boot_neglogpdist = np.zeros((n_iter, bins.size-1))
boot_neglogpdist_N = np.zeros((n_iter, bins.size-1))

for i, (this_logweights, boot_data, boot_data_N) in enumerate(dat):

    this_neglogpdist, this_neglogpdist_N, this_avg, this_chi, this_avg_N, this_chi_N, _ = extract_and_reweight_data(this_logweights, boot_data, boot_data_N, bins, beta_phi_vals)

    boot_neglogpdist[i] = this_neglogpdist
    boot_neglogpdist_N[i] = this_neglogpdist_N
    boot_avg_phi[i] = this_avg
    boot_chi_phi[i] = this_chi



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


## FIND SMOOTHED SUSCEPTIBILITIES ##
boot_smooth_chi = np.zeros((n_iter, beta_phi_vals.size))
boot_peak_sus = np.zeros(n_iter)
## Now that we've determined the std error in N v phi, we can fit each to a 
#    smoothing spline to determine beta phi star
for i, avg_n_v_phi in enumerate(boot_avg_phi):
    spl = scipy.interpolate.UnivariateSpline(beta_phi_vals, avg_n_v_phi, w=1/err_tot_N)
    smooth_chi = -spl(beta_phi_vals, 1)

    this_peak_sus = beta_phi_vals[np.argmax(smooth_chi)]

    boot_smooth_chi[i] = smooth_chi
    boot_peak_sus[i] = this_peak_sus

spl = scipy.interpolate.UnivariateSpline(beta_phi_vals, all_avg, w=1/err_tot_N)
all_smooth_chi = -spl(beta_phi_vals, 1)

tot_smooth_chi = boot_smooth_chi.mean(axis=0)
err_smooth_chi = boot_smooth_chi.std(axis=0, ddof=1)
avg_peak_sus = boot_peak_sus.mean()
err_peak_sus = boot_peak_sus.std(ddof=1)

dat = np.vstack((beta_phi_vals, all_avg, err_tot_N, all_smooth_chi, err_smooth_chi, all_chi, err_tot_chi)).T
np.savetxt('NvPhi.dat', dat, header='beta*phi   <N_v>  err(<N_v>)  smooth_chi_v err(smooth_chi_v) chi_v   err(chi_v) ')

np.savetxt('peak_sus.dat', np.array([avg_peak_sus, err_peak_sus]), header='beta*phi^*  err(beta phi*)')


print('...Done.')

