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


from constants import k

fig, axes = plt.subplots(2, 3, sharex=True, figsize=(20,10))
axes = axes.flatten()
beta = 1/(k*300)

fnames = sorted( glob.glob('*/phi_sims/out.dat') )

for idir, fname in enumerate(fnames):
    ax = axes[idir]
    dirname = os.path.dirname(fname)
    dat = np.loadtxt(fname)
    var_err_dat = np.loadtxt('{}/ntwid_var_err.dat'.format(dirname))

    ax.errorbar(dat[:,0], dat[:,-1], yerr=var_err_dat, fmt='k-o', linewidth=6, elinewidth=3)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(0, xmax)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)

#plt.tight_layout()
plt.show()