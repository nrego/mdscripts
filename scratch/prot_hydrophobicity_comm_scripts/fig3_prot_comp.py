from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import glob, os

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':30})

name_lup = {'1brs': 'barnase',
            '1ubq': 'ubiquitin',
            '1qgt': 'capsid',
            '1ycr': 'mdm2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate dehydrogenase'}

order = ['hydrophobin', 'capsid', 'lysozyme', 'mdm2', 'malate dehydrogenase', 'barnase']

from constants import k

fig, axes = plt.subplots(2, 3, sharex=True, figsize=(20,10))
axes = axes.flatten()
beta = 1/(k*300)

order_idx = []
fnames = np.array([], dtype=str)
for key, val in name_lup.iteritems():
    if val in order:
        order_idx.append(order.index(val))
        fnames = np.append(fnames, '{}/phi_sims/out.dat'.format(key))
fnames = fnames[np.argsort(order_idx)]

for idir, fname in enumerate(fnames):
    ax = axes[idir]
    dirname = os.path.dirname(fname)
    dat = np.loadtxt(fname)

    avg_err_dat = np.loadtxt('{}/ntwid_err.dat'.format(dirname))
    var_err_dat = np.loadtxt('{}/ntwid_var_err.dat'.format(dirname))

    #ax.errorbar(dat[:,0], dat[:,-1], yerr=var_err_dat, fmt='k-o', linewidth=6, elinewidth=3)
    ax.errorbar(dat[:,0], dat[:,1], yerr=avg_err_dat, fmt='k-o', linewidth=6, elinewidth=3, markersize=12)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(0, beta*10)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)
    ax.set_title(order[idir])
    

plt.tight_layout()
plt.show()

