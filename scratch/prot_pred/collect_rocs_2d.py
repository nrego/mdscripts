# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from constants import k

import numpy as np

import argparse
mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

beta = 1 /(k*300)

fnames = glob.glob('s_*/beta_phi_*/pred_contact_mask.dat')

buried_mask = np.loadtxt('../bound/buried_mask.dat', dtype=bool)
surf_mask = ~buried_mask

contacts = np.loadtxt('../bound/actual_contact_mask.dat', dtype=bool)

beta_phi_vals = np.unique([float(fname.split('/')[1].split('_')[-1])/100.0  for fname in fnames])
s_vals = np.unique([float(fname.split('/')[0].split('_')[-1])/100.0 for fname in fnames])

f1_vals = np.zeros((len(beta_phi_vals), len(s_vals)), dtype=float)
d_h_vals = np.zeros((len(beta_phi_vals), len(s_vals)), dtype=float)
d_vals = np.zeros((len(beta_phi_vals), len(s_vals)), dtype=float)

for fname in fnames:
    pred = np.loadtxt(fname, dtype=bool)

    tp = (contacts[surf_mask] & pred[surf_mask]).sum()
    fp = (~contacts[surf_mask] & pred[surf_mask]).sum()
    tn = (~contacts[surf_mask] & ~pred[surf_mask]).sum()
    fn = (contacts[surf_mask] & ~pred[surf_mask]).sum()

    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)

    prec = tp/(tp+fp)

    f1 = 2*tp/(2*tp + fp + fn)

    d_h = 2 /((1/tpr) + (1/(1-fpr)) )
    d = np.sqrt((1-tpr)**2 + (fpr)**2)

    this_phi = float(fname.split('/')[1].split('_')[-1])/100.0
    this_s = float(fname.split('/')[0].split('_')[-1])/100.0

    phi_bin_idx = np.where(beta_phi_vals==this_phi)[0][0]
    s_bin_idx = np.where(s_vals==this_s)[0][0]

    f1_vals[phi_bin_idx,s_bin_idx] = f1
    d_h_vals[phi_bin_idx,s_bin_idx] = d_h
    d_vals[phi_bin_idx,s_bin_idx] = d

fig, ax = plt.subplots(figsize=(9,7))

vals_to_plot = d_h_vals

norm = Normalize(0.0,0.8)
im = ax.imshow(vals_to_plot.T, aspect='auto', origin='bottom', extent=(0,4.04,0,1.1), norm=norm)
ax.axhline(0.5, 0, 4.04, color='r')
ax.axhline(0.6, 0, 4.04, color='r')
ax.axvline(2.24, 0, 1.1, color='r')
ax.axvline(2.28, 0, 1.1, color='r')
fig.colorbar(im)
ax.set_yticks(np.arange(0,1.1,0.1))
ax.set_xticks(np.arange(0,4.1,1))
ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$s$')
fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/2droc.pdf', transparent=True)

plt.close('all')

mpl.rcParams.update({'axes.labelsize': 60})
mpl.rcParams.update({'xtick.labelsize': 38})
mpl.rcParams.update({'ytick.labelsize': 38})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})
## Plot d_h v. beta phi at s=0.5
idx=5
assert s_vals[idx] == 0.5
dat = d_h_vals[:,idx]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13,5.75), sharey=True)
ax1.plot(beta_phi_vals, dat, 'ko', markersize=12)
ax1.set_yticks(np.arange(0,0.9,0.1))
ymin, ymax = ax1.get_ylim()
ax1.set_ylim(ymin, 0.8)
ax1.set_xticks([0,2,4])


## Plot d_h v. beta phi at s=0.5
idx = np.argmax(dat)
assert beta_phi_vals[idx] == 2.24
dat = d_h_vals[idx, :]

ax2.plot(10*s_vals, dat, 'ko', markersize=12)
ax2.set_xticks([0,5,10])

#for ax in fig.axes:
#    ax.label_outer()
ax2.tick_params(left=False)
fig.subplots_adjust(wspace=.2)
fig.savefig('/Users/nickrego/Desktop/d_v_phi_and_s.pdf', transparent=True)


