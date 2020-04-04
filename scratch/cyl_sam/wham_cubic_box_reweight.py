
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

boot_indices = np.load('boot_indices.dat.npy')
dat = np.load('boot_fn_payload.dat.npy')
n_iter = boot_indices.shape[0]

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)
bins = np.arange(0, max_val+1, 1)

## In kT!
beta_phi_vals = np.arange(0,6.02,0.02)

## EXTRACT DATA TO REWEIGHT ##
### Extract all N_V's (number of waters in cube vol V)
print('')
print('Extracting cubic probe volume data\'s...')
sys.stdout.flush()

# Number of waters in V for each frame of each traj - collect them
cube_dat_fnames = sorted(glob.glob("Nstar_*/phiout_cube.dat"))

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

# Save out the average 
avg_com = np.array([all_avg_comx[0], all_avg_comy[0], all_avg_comz[0]])
np.savez_compressed("cube_data_equil.dat", avg_com=avg_com, n0=all_avg_cube[0])


## If 2d density profiles are ready - reweight them, too, to find the unbiased rho(z,r) ###

# Now check to see if rho_z data is ready
all_data_rhoz = []
rhoz_dat_fnames = sorted(glob.glob("Nstar_*/rhoz.dat.npz"))
if len(rhoz_dat_fnames) > 0:
    print("Doing rhoz...")

xvals = None
rvals = None
for fname in rhoz_dat_fnames:

    ds = np.load(fname)

    if xvals is None:
        xvals = ds['xvals']
    if rvals is None:
        rvals = ds['rvals']

    assert np.array_equal(ds['xvals'], xvals)
    assert np.array_equal(ds['rvals'], rvals)

    this_rhoz = ds['rho_z']
    all_data_rhoz.append(this_rhoz)

all_data_rhoz = np.concatenate(all_data_rhoz, axis=0)
n_tot, nx, nr = all_data_rhoz.shape

# Flatten last dimension
#. Shape: (n_tot_data, n_cyls)
all_data_rhoz = all_data_rhoz.reshape(n_tot, -1)


# Now find density profile for all bphi vals
rr, xx = np.meshgrid(rvals[:-1], xvals[:-1], indexing='ij')
#xx -= xx.min()


all_rho = np.zeros((beta_phi_vals.size, nx, nr))

# Should really abstract back to method (TODO), but can do later
for i, beta_phi_val in enumerate(beta_phi_vals):
    print("  doing {} of {}".format(i, beta_phi_vals.size))
    bias_logweights = all_logweights - beta_phi_val*all_data
    bias_logweights -= bias_logweights.max()
    #norm = np.log(math.fsum(np.exp(bias_logweights)))
    norm = np.log(np.sum(np.exp(bias_logweights)))
    bias_logweights -= norm

    bias_weights = np.exp(bias_logweights)

    this_rho = np.dot(bias_weights, all_data_rhoz).reshape(nx, nr)

    all_rho[i] = this_rho

# Save total density profile
np.savez_compressed('rhoz_final.dat', rhoz=all_rho, beta_phi_vals=beta_phi_vals, 
                    xvals=xvals, rvals=rvals, xx=xx, rr=rr)



