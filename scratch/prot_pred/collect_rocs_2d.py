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

from constants import k

import numpy as np

import argparse

beta = 1 /(k*300)

fnames = glob.glob('s_*/phi_*/pred_contact_mask.dat')

buried_mask = np.loadtxt('../bound/buried_mask.dat', dtype=bool)
surf_mask = ~buried_mask

contacts = np.loadtxt('../bound/actual_contact_mask.dat', dtype=bool)

phi_vals = np.unique([float(fname.split('/')[1].split('_')[-1])/10.0  for fname in fnames])
s_vals = np.unique([float(fname.split('/')[0].split('_')[-1])/100.0 for fname in fnames])

f1_vals = np.zeros((len(phi_vals), len(s_vals)), dtype=float)
d_h_vals = np.zeros((len(phi_vals), len(s_vals)), dtype=float)
d_vals = np.zeros((len(phi_vals), len(s_vals)), dtype=float)

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

    this_phi = float(fname.split('/')[1].split('_')[-1])/10.0
    this_s = float(fname.split('/')[0].split('_')[-1])/100.0

    phi_bin_idx = np.where(phi_vals==this_phi)[0][0]
    s_bin_idx = np.where(s_vals==this_s)[0][0]

    f1_vals[phi_bin_idx,s_bin_idx] = f1
    d_h_vals[phi_bin_idx,s_bin_idx] = d_h
    d_vals[phi_bin_idx,s_bin_idx] = d

phi_bins = np.append(phi_vals, phi_vals[-1]+1)
s_bins = np.append(s_vals, s_vals[-1]+0.1)

fig, ax = plt.subplots(figsize=(15,10))

vals_to_plot = f1_vals
#norm = Normalize(vals_to_plot.min(), vals_to_plot.max())
norm = Normalize(0.4, 0.54)
im = ax.pcolormesh(beta*phi_bins, s_bins, vals_to_plot.T, norm=norm)
fig.colorbar(im)
ax.set_yticks(np.arange(0,1.1,0.1))
ax.set_xticks(np.arange(0,4.1,1))
ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$s$')
fig.tight_layout()
plt.show()




