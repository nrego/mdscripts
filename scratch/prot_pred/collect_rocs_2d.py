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

fig, ax1 = plt.subplots()
ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

ideal_pt = np.array([0, 1])

fnames = sorted(glob.glob('phi_*roc.dat'))

roc_dat = np.loadtxt(fnames[0])
thresholds = roc_dat[:,0]

roc_2d = np.zeros((len(fnames), thresholds.size))
#plt.plot([0], [1], '*', markersize=10)

phi_vals = []

for i, fname in enumerate(fnames):
    phi_val = float(fname.split('_')[1]) / 10.
    phi_vals.append(phi_val)

    roc_dat = np.loadtxt(fname)

    dists = np.abs(ideal_pt - roc_dat[:,1:])

    l1norm = np.linalg.norm(dists, ord=1, axis=1)
    l2norm = np.linalg.norm(dists, ord=2, axis=1)

    min_idx = 9

    roc_2d[i,:] = l2norm

del phi_val
phi_vals = np.array(phi_vals)

norm = Normalize(vmin=0.0, vmax=1, clip=False)
im = ax1.imshow(roc_2d.T, interpolation='none', origin='lower', norm=norm, aspect='auto', cmap=cm.hot)
cb = plt.colorbar(im)
ax1.set_xticks(np.arange(phi_vals.size)[::2])
ax1.set_xticklabels(np.round(phi_vals[::2]*beta, 1))
ax1.set_yticks(np.arange(thresholds.size)[::2])
ax1.set_yticklabels(thresholds[::2])
np.savetxt('roc_dist_with_phi.dat', roc_2d[:,10])
ax1.set_ylabel(r'$s$')
ax1.set_xlabel(r'$\beta \phi$')
plt.show()

#fig, ax1 = plt.subplots()
#ax1.plot(beta*phi_vals, roc_2d[:,10], '-ok', linewidth=8, markersize=18)
ax1.set_xlabel(r'$\beta \phi$')
ax1.set_ylabel(r'$d$')
#plt.show()

#fig, ax1 = plt.subplots()
#ax1.plot(thresholds, roc_2d[6,:], '-ok', linewidth=8, markersize=18)
ax1.set_xlabel(r'$s$')
ax1.set_ylabel(r'$d$')
#plt.show()


