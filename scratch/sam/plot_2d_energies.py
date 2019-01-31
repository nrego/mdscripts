from __future__ import division, print_function

import numpy as np
import glob, os

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from IPython import embed
## From pattern_sample directory, extract free energies for each k, d value
mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 10})
mpl.rcParams.update({'ytick.labelsize': 10})
mpl.rcParams.update({'axes.titlesize': 30})

max_val = 1.75
rms_bins = np.arange(0, max_val+0.1, 0.05, dtype=np.float32)


fnames = glob.glob('k_*/d_*/trial_0/PvN.dat') + ['k_36/PvN.dat', 'k_00/PvN.dat']
#fnames = glob.glob('l_*/d_*/trial_0/PvN.dat')

k_bins = np.sort(np.unique([int(fname.split('/')[0].split('_')[-1]) for fname in fnames]))
k_bins = np.append(k_bins, k_bins.max()+1)
k_bins = k_bins.astype(np.float32)

energies = np.zeros((rms_bins.size-1, k_bins.size-1))
energies[:] = np.nan
errors = np.zeros((rms_bins.size-1, k_bins.size-1))
errors[:] = np.nan
range_k = np.zeros(k_bins.size-1)

for fname in fnames:

    this_k = float(fname.split('/')[0].split('_')[-1])
    
    if this_k == 36:
        this_d = 1.14
    elif this_k == 0:
        this_d = 0
    else:
        this_d = float(fname.split('/')[1].split('_')[-1]) / 100


    idx_k = np.digitize(this_k, k_bins) - 1
    idx_d = np.digitize(this_d, rms_bins) - 1

    this_fvn = np.loadtxt(fname)

    this_mu, this_err = this_fvn[0, 1:]

    energies[idx_d, idx_k] = this_mu
    errors[idx_d, idx_k] = this_err

for idx_k in range(k_bins.size-1):
    range_k[idx_k] = np.nanmax(energies[:,idx_k]) - np.nanmin(energies[:,idx_k])

fig, ax = plt.subplots(figsize=(6,5))
ax.bar(k_bins[:-1], range_k, width=0.8, align='edge')
ax.set_ylabel(r'$\Delta \beta F$')
ax.set_xlabel(r'$k$')
fig.savefig('/Users/nickrego/Desktop/range_k.pdf')
plt.close('all')

fig, ax = plt.subplots(figsize=(6,5))

min_energy = np.nanmin(energies)
max_energy = np.nanmax(energies)
norm = mpl.colors.Normalize(vmin=min_energy, vmax=max_energy)

extent = (rms_bins[0], rms_bins[-1], k_bins[0], k_bins[-1])
im = ax.imshow(energies.T, extent=extent, origin='lower', aspect='auto', norm=norm, cmap=cm.nipy_spectral)
cb = plt.colorbar(im)
ax.set_xlabel(r'$d$ (RMS)')
ax.set_ylabel(r'$k$')

fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/rms_k_2d.pdf')

fig, ax = plt.subplots(figsize=(6,5))

min_err = np.nanmin(errors)
max_err = np.nanmax(errors)
norm = mpl.colors.Normalize(vmin=min_err, vmax=max_err)

extent = (rms_bins[0], rms_bins[-1], k_bins[0], k_bins[-1])
im = ax.imshow(errors.T, extent=extent, origin='lower', aspect='auto', norm=norm, cmap=cm.nipy_spectral)
cb = plt.colorbar(im)

fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/err.pdf')