from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import matplotlib as mpl
from matplotlib import cm
import os, glob


max_val = 1.75
rms_bins = np.arange(0, max_val+0.1, 0.05, dtype=np.float32)

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 10})
mpl.rcParams.update({'ytick.labelsize': 10})
mpl.rcParams.update({'axes.titlesize': 30})

fnames = glob.glob('k_*/d_*/trial_0/act_diff.dat')
fnames.append('k_00/act_diff.dat')
fnames.append('k_36/act_diff.dat')

this_ks = np.arange(36)
k_bins = np.sort(np.unique([int(fname.split('/')[0].split('_')[-1]) for fname in fnames]))
k_bins = np.append(k_bins, k_bins.max()+1)
k_bins = k_bins.astype(np.float32)

energies = np.zeros((rms_bins.size-1, k_bins.size-1))
energies[:] = np.nan

for fname in fnames:
    this_k = int(fname.split('/')[0].split('_')[-1])
    if this_k == 0:
        this_d = np.nan
    elif this_k == 36:
        this_d = 1.14
    else:
        this_d = float(fname.split('/')[1].split('_')[-1]) / 100

    dat = float(np.loadtxt(fname))

    print("k: {}  d: {}   dg: {}".format(this_k, this_d, dat))

    idx_k = np.digitize(this_k, k_bins) - 1
    idx_d = np.digitize(this_d, rms_bins) - 1

    try:
        energies[idx_d, idx_k] = dat
    except:
        pass

fig, ax = plt.subplots(figsize=(6,5))

lim_energy = np.nanmax(np.abs(energies))

norm = mpl.colors.Normalize(vmin=-lim_energy, vmax=lim_energy)

extent = (rms_bins[0], rms_bins[-1], k_bins[0], k_bins[-1])
im = ax.imshow(energies.T, extent=extent, origin='lower', aspect='auto', norm=norm, cmap=cm.seismic_r)
cb = plt.colorbar(im)
ax.set_xlabel(r'$d$ (RMS)')
ax.set_ylabel(r'$k$')
#ax.set_yticks(np.arange(0,37,4))

fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/blah.pdf')

