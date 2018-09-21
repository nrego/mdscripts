from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import sys, os

homedir = os.environ['HOME']

from constants import k

# Some shennanigans to import when running from IPython terminal
try:
    from utils import plt_errorbars
except:
    import imp
    utils = imp.load_source('utils', '{}/mdscripts/scratch/prot_hydrophobicity_comm_scripts/utils.py'.format(homedir))
    plt_errorbars = utils.plt_errorbars

mpl.rcParams.update({'axes.labelsize': 100})
mpl.rcParams.update({'xtick.labelsize': 80})
mpl.rcParams.update({'ytick.labelsize': 80})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':30})

fig, ax = plt.subplots(figsize=(8.5,7))
beta = 1#/(k*300)
dat = np.loadtxt('n_v_phi.dat')
err_dat = dat[:,2]

ax.plot(dat[:,0], dat[:,1], 'k-', linewidth=6)
plt_errorbars(dat[:,0], dat[:,1], dat[:,2])

ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$\langle N_v \rangle_{\phi}$')

ax.set_xlim(0, 4)

fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/ubiq_n_v_phi.pdf', transparent=True)
fig.show()



fig, ax = plt.subplots(figsize=(8.325,7))

dat = np.loadtxt('var_n_v_phi.dat')
ax.plot(dat[:,0], dat[:,1], 'k-', linewidth=6)
plt_errorbars(dat[:,0], dat[:,1], dat[:,2], color='k')

ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$\chi_v$')

ax.set_xlim(0, 4)


fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/ubiq_sus_v_phi.pdf', transparent=True)
fig.show()


