
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

def plot_errorbar(bb, dat, err, **kwargs):
    plt.plot(bb, dat, **kwargs)
    plt.fill_between(bb, dat-err, dat+err, alpha=0.5)

def bootstrap_nvphi(boot_payload, beta_phi_vals, bins):

    n_boot = boot_payload.shape[0]
    out = np.zeros((n_boot, beta_phi_vals.size))

    for i, (boot_logweights, boot_data, boot_data_N) in enumerate(boot_payload):
        neglogpdist, neglogpdist_N, avg, chi, avg_N, chi_N, cov_N = extract_and_reweight_data(boot_logweights, boot_data, boot_data_N, bins, beta_phi_vals)
        print(i)
        out[i] = chi


    return out

print('Constructing Nv v phi, chi v phi...')
sys.stdout.flush()
temp = 300

beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})


### EXTRACT MBAR DATA ###

all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_aux']

#boot_indices = np.load('boot_indices.dat.npy')
#dat = np.load('boot_fn_payload.dat.npy')
#n_iter = boot_indices.shape[0]

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)
bins = np.arange(0, max_val+1, 1).astype(int)

## In kT!
beta_phi_vals = np.arange(0,6.02,0.02)
beta_phi_vals = np.unique(np.round(np.append(beta_phi_vals, np.arange(0.4, 0.5, 0.005)), 4))

## EXTRACT DATA TO REWEIGHT ##
### Extract all N_V's (number of waters in cube vol V)
print('')
print('Extracting cubic probe volume data\'s...')
sys.stdout.flush()

# Number of waters in V for each frame of each traj - collect them
cube_dat_fnames = sorted(glob.glob("*/phiout_cube.dat"))

# Final shape: (n_total_data,)
# Number of waters in cubic probe volume V, all frames from all simulations
all_data_cube = []
all_data_com = []

## Gather n_i data from each umbrella window (nstar value)
for fname in cube_dat_fnames:
    dirname = os.path.dirname(fname)
    all_data_cube.append(np.loadtxt(fname))
    all_data_com.append(np.load('{}/com_cube.dat.npy'.format(dirname)))

all_data_cube = np.concatenate(all_data_cube)
all_data_com = np.concatenate(all_data_com, axis=0)

print('')

print('WHAMing and reweighting to phi-ens')
sys.stdout.flush()

# Make sure our PV(N) goes to high enough N
bins = np.arange(all_data_cube.max()+3)

# Find <N_V>0 and unbiased average water COM in V
all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_cube, all_chi_N, all_cov_cube = extract_and_reweight_data(all_logweights, all_data, all_data_cube, bins, beta_phi_vals)
all_neglogpdist, all_neglogpdist_comx, all_avg, all_chi, all_avg_comx, all_chi_comx, all_cov_comx = extract_and_reweight_data(all_logweights, all_data, all_data_com[:,0], bins, beta_phi_vals)
all_neglogpdist, all_neglogpdist_comy, all_avg, all_chi, all_avg_comy, all_chi_comy, all_cov_comy = extract_and_reweight_data(all_logweights, all_data, all_data_com[:,1], bins, beta_phi_vals)
all_neglogpdist, all_neglogpdist_comz, all_avg, all_chi, all_avg_comz, all_chi_comz, all_cov_comz = extract_and_reweight_data(all_logweights, all_data, all_data_com[:,2], bins, beta_phi_vals)

all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_N, all_chi_N, all_cov_N = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)

this_dat = np.vstack((beta_phi_vals, all_avg, all_chi)).T
np.savetxt("NvPhi.dat", this_dat)


# Find chi_v max! (small probe)
max_idx = np.argmax(all_chi)
bphistar = beta_phi_vals[max_idx]
print("beta phi star: {:.2f}".format(bphistar))
print("<N>_phistar: {:.2f}".format(all_avg[max_idx]))
plt.plot(beta_phi_vals, all_chi)

# Save out the average 

avg_com = np.array([all_avg_comx[0], all_avg_comy[0], all_avg_comz[0]])
print("Average com: {}".format(avg_com))
print("<N>_0: {:.2f}".format(all_avg_cube[0]))
np.savez_compressed("cube_data_equil.dat", avg_com=avg_com, n0=all_avg_cube[0], chi0=all_chi_N[0])

