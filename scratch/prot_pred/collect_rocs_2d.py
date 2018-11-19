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



ideal_pt = np.array([0, 1])

fnames = sorted(glob.glob('thresh*/phi_*/accuracy.dat'))

thresholds = [fname.split('/')[0].split('_')[-1] for fname in fnames]
thresholds = np.unique(thresholds).astype(float) / 10.0
#thresholds = np.append(thresholds, 1.1)

phi_vals = [fname.split('/')[1].split('_')[-1] for fname in fnames]
phi_vals = np.unique(phi_vals).astype(float) / 10.0

roc_2d = np.zeros((phi_vals.size, thresholds.size, 2))
dist = np.zeros((phi_vals.size, thresholds.size))
roc_2d[...] = np.nan
dist[...] = np.nan

for fname in fnames:
    tp, fp, tn, fn = np.loadtxt(fname)
    this_phi = float(fname.split('/')[1].split('_')[-1]) / 10.
    this_thresh = float(fname.split('/')[0].split('_')[-1]) / 10.

    phi_idx = np.argwhere(phi_vals == this_phi)[0,0]
    thresh_idx = np.argwhere(thresholds == this_thresh)[0,0]

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)

    roc_2d[phi_idx, thresh_idx] =  fpr, tpr
    this_dist = np.sqrt((tpr - 1)**2 + fpr**2)
    dist[phi_idx, thresh_idx] = this_dist

fig, ax1 = plt.subplots()
ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

norm = Normalize(vmin=0.30, vmax=1, clip=False)
#im = ax1.imshow(dist.T, interpolation='none', origin='lower', norm=norm, aspect='auto', cmap=cm.hot)
im = ax1.pcolormesh(np.append(phi_vals, 11.0), np.append(thresholds, 1.1), dist.T, norm=norm, cmap=cm.hot)
cb = plt.colorbar(im)

plt.show()



