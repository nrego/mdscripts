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

roc_2d = np.zeros((len(fnames), thresholds.size, 2))
#plt.plot([0], [1], '*', markersize=10)

phi_vals = []

for i, fname in enumerate(fnames):
    phi_val = float(fname.split('_')[1]) / 10.
    phi_vals.append(phi_val)

    roc_dat = np.loadtxt(fname)

    roc_2d[i,:, ...] = roc_dat[:,1:]

del phi_val
phi_vals = np.array(phi_vals)

phi_idx = 6 # phi=4.5
thresh_idx = 10 # s=0.5

phi_roc = roc_2d[phi_idx, :]
thresh_roc = roc_2d[:, thresh_idx, :]

fig, ax = plt.subplots()
ax.plot(0,1,'*', markersize=20, color='r')

np.hstack((phi_vals[:,None],thresh_roc))
ax.plot(phi_roc[:,0], phi_roc[:,1], '-ok', markersize=18, linewidth=6)
ax.axhline(1, linestyle='--', color='k')
ax.axvline(0, linestyle='--', color='k')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')


