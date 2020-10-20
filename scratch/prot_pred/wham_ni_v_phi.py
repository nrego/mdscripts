
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

from whamutils import get_negloghist, extract_and_reweight_data
from work_managers.environment import default_env
import argparse


parser = argparse.ArgumentParser("reweight to get ni v phi")
parser.add_argument('--min-bphi', type=float, default=0.0)
parser.add_argument('--max-bphi', type=float, default=4.0)
parser.add_argument('--n-vals', type=int, default=101,
                    help='number of bphi vals between 0 and bphimax (default: 101)')
parser.add_argument('--do-smooth', action='store_true', help='If true, smooth chi curves (default: False)')
default_env.add_wm_args(parser)

args = parser.parse_args()
default_env.process_wm_args(args)
wm = default_env.make_work_manager()
wm.startup()
print("wm {} started with {} workers".format(wm, wm.n_workers))

## Construct -ln P_v(N) from wham results (after running whamerr.py with '--boot-fn utility_functions.get_weighted_data')
## also get <N> v phi, and suscept

do_smooth = False

def plot_errorbar(bb, dat, err):
    plt.plot(bb, dat)
    plt.fill_between(bb, dat-err, dat+err, alpha=0.5)


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
beta_phi_vals = np.linspace(args.min_bphi, args.max_bphi, args.n_vals)
print("bphi vals: {}".format(beta_phi_vals))

## Get PvN, <Nv>, chi_v from all data ###
all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_N, all_chi_N, _ = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)

### Now input all <n_i>_\phi's for a given i ###
print('')
print('Extracting all n_i\'s...')
sys.stdout.flush()

n_heavies = None

n_i_dat_fnames = np.append(sorted(glob.glob('phi*/rho_data_dump_rad_6.0.dat.npz')), sorted(glob.glob('nstar*/rho_data_dump_rad_6.0.dat.npz')))
# Final shape: (n_heavies, n_total_data)
all_data_n_i = None

## Gather n_i data from each umbrella window (phi value)
for fname in n_i_dat_fnames:

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
sys.stdout.flush()

# Shape: (n_atoms, beta_phi_vals.shape)
avg_nis = np.zeros((n_heavies, beta_phi_vals.size))
smooth_avg_nis = np.zeros_like(avg_nis)
chi_nis = np.zeros((n_heavies, beta_phi_vals.size))
cov_nis = np.zeros((n_heavies, beta_phi_vals.size))
smooth_cov_nis = np.zeros_like(cov_nis)

futures = []

def task_gen():
    for i_atm in range(n_heavies):

        import time
        start = time.time()

        #neglogpdist, neglogpdist_ni, avg, chi, avg_ni, chi_ni, cov_ni = extract_and_reweight_data(all_logweights, all_data, all_data_n_i[i_atm], bins, beta_phi_vals)

        fn_args = (all_logweights, all_data, all_data_n_i[i_atm], bins, beta_phi_vals)
        fn_kwargs = {'this_idx': i_atm}
        
        #futures.append(wm.submit(get_rhoxyz, fn_args, fn_kwargs))
        #futures.append(wm.submit(extract_and_reweight_data, fn_args, fn_kwargs))
        print("sending job for atm {}".format(i_atm))
        yield (extract_and_reweight_data, fn_args, fn_kwargs)

        #print("submitted job {}".format(i))

        #if do_smooth:
        #    boot_avg_ni = np.zeros((n_iter, beta_phi_vals.size))
        #    for i_boot in range(n_iter):
        #        this_boot_indices = boot_indices[i_boot]
        #        (this_logweights, boot_data, boot_data_N) = dat[i_boot]

        #        _, _, _, _, this_boot_avg_ni, _, _ = extract_and_reweight_data(this_logweights, boot_data, all_data_n_i[i_atm][boot_indices[i_boot]], bins, beta_phi_vals)
        #        boot_avg_ni[i_boot] = this_boot_avg_ni

        #    err_avg_ni = np.ma.masked_invalid(boot_avg_ni).std(axis=0, ddof=1)
        #    spl = scipy.interpolate.UnivariateSpline(beta_phi_vals, avg_ni, w=1/err_avg_ni)
            
        #    smooth_avg_nis[i_atm, :] = spl(beta_phi_vals)
        #    smooth_cov_nis[i_atm, :] = -spl(beta_phi_vals, 1)

        #avg_nis[i_atm, :] = avg_ni
        #chi_nis[i_atm, :] = chi_ni
        #cov_nis[i_atm, :] = cov_ni


        #end = time.time()

        #if i_atm % 1 == 0:
        #    print('  i: {}, ({:.2f}s)'.format(i_atm, end-start), end='\r')
        #    sys.stdout.flush()

## COllect results
for i, future in enumerate(wm.submit_as_completed(task_gen(), 4)):
    i_atm, neglogpdist, neglogpdist_ni, avg, chi, avg_ni, chi_ni, cov_ni = future.get_result(discard=True)
    if i % 1 == 0:
        print("getting result {} of {}".format(i, n_heavies))
        sys.stdout.flush()
    
    avg_nis[i_atm, :] = avg_ni
    chi_nis[i_atm, :] = chi_ni
    cov_nis[i_atm, :] = cov_ni

wm.shutdown()

np.savez_compressed('ni_rad_weighted.dat', avg=avg_nis, var=chi_nis, cov=cov_nis, smooth_avg=smooth_avg_nis, smooth_cov=smooth_cov_nis, beta_phi=beta_phi_vals)

