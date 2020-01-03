
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
import MDAnalysis
from whamutils import get_negloghist, extract_and_reweight_data
import pymbar

def subsample(data, n_sample, uncorr_n_sample):
    np.random.seed()

    sub_data = np.zeros(uncorr_n_sample)
    avail_indices = np.arange(n_sample)
    sub_indices = np.random.choice(avail_indices, size=uncorr_n_sample, replace=True)

    return data[sub_indices]

def bootstrap_single(data, n_sample, uncorr_n_sample, n_iter=64):
    boot_data = np.zeros((n_iter, uncorr_n_sample))
    for i in range(n_iter):
        boot_data[i] = subsample(data, n_sample, uncorr_n_sample)

    return boot_data

#local (i.e. surf) index
local_idx = 1331


beta = 1 / (300 * k)
univ = MDAnalysis.Universe('bound/actual_contact.pdb')
buried_mask = np.loadtxt('bound/buried_mask.dat', dtype=bool)
surf_mask = ~buried_mask
indices = np.arange(univ.atoms.n_atoms)
# Lookup local surf atom index to global heavy index
surf_indices = indices[surf_mask]

idx = surf_indices[local_idx]
# Plot smoothed and unsmoothed for a given n_i


# Filenames in of the actual (not whammed) <n>'s'
get_float = lambda fname: beta * float(fname.split("/")[2].split("_")[1]) / 10.0

fnames = sorted(glob.glob("phi_sims/data_reduced/phi_*/rho_data_dump_rad_6.0.dat.npz"))
file_beta_phi = np.zeros(len(fnames))
file_ni = np.zeros_like(file_beta_phi)
file_ni_std = np.zeros_like(file_ni)

all_data_n_i = None
n_samples = np.array([], dtype=int)
uncorr_n_samples = np.array([], dtype=int)

for i, fname in enumerate(fnames):
    file_beta_phi[i] = get_float(fname)
    n_i = np.load(fname)['rho_water'].T[idx]
    file_ni[i] = n_i.mean()
    try:
        tau = pymbar.timeseries.integratedAutocorrelationTime(n_i)
        uncorr_n_samples = np.append(uncorr_n_samples, n_i.size // (1+2*tau)).astype(int)
    except:
        uncorr_n_samples = np.append(uncorr_n_samples, 1)

    n_samples = np.append(n_samples, n_i.size).astype(int)
    
    boot_ni = bootstrap_single(n_i, n_samples[i], uncorr_n_samples[i])
    file_ni_std[i] = boot_ni.mean(axis=1).std(ddof=1)

    if all_data_n_i is None:
        all_data_n_i = n_i.copy()
    else:
        all_data_n_i = np.append(all_data_n_i, n_i)

fnames = sorted(glob.glob("phi_sims/data_reduced/nstar_*/rho_data_dump_rad_6.0.dat.npz"))
for fname in fnames:
    n_i = np.load(fname)['rho_water'].T[idx]
    all_data_n_i = np.append(all_data_n_i, n_i)

all_data_ds = np.load('phi_sims/data_reduced/all_data.dat.npz')
all_logweights = all_data_ds['logweights']
assert all_data_n_i.size == all_logweights.size

# Now get continuous (whammed') and smoothed curves
ds = np.load('phi_sims/ni_rad_weighted.dat.npz')
wham_ni = ds['avg'][idx]
smooth_ni = ds['smooth_avg'][idx]
wham_cov = ds['cov'][idx]
smooth_cov = ds['smooth_cov'][idx]
beta_phi = ds['beta_phi']

# Now estimate errors on wham_ni from bootstraps (check if smoothing is working correctly)
boot_indices = np.load("phi_sims/data_reduced/boot_indices.dat.npy")
n_iter = boot_indices.shape[0]

boot_ni

ax1 = plt.gca()
ax1.plot(beta_phi, wham_ni, label='wham')
ax1.plot(beta_phi, smooth_ni, 'k--', label='smooth')
ax1.errorbar(file_beta_phi, file_ni, yerr=file_ni_std, fmt='o', label='sim avg')
ax1.legend()
plt.show()

ax2 = plt.gca()
ax2.plot(beta_phi, wham_cov)
ax2.plot(beta_phi, smooth_cov, '-', color='orange', label='smooth')
ax2.legend()
plt.show()