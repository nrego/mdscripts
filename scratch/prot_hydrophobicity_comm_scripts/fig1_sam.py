from __future__ import division, print_function
  
import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import sys, os

from constants import k

homedir = os.environ['HOME']

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

ch3_dat = np.loadtxt('CH3_Disks/n_v_phi.dat')
oh_dat = np.loadtxt('OH_Disks/n_v_phi.dat')

# N v phi
fig, ax = plt.subplots(figsize=(8.5,7))

ax.plot(ch3_dat[:,0], ch3_dat[:,1], 'r-', label=r'$\rm{CH}_3$', linewidth=6)
plt_errorbars(ch3_dat[:,0], ch3_dat[:,1], ch3_dat[:,2])
ax.plot(oh_dat[:,0], oh_dat[:,1], 'b-', label=r'$\rm{OH}$', linewidth=6)
plt_errorbars(oh_dat[:,0], oh_dat[:,1], oh_dat[:,2])

ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$\langle N_v \rangle_{\phi}$')

ax.set_xlim(0,2)
ax.set_xticks([0,1,2])
ax.set_ylim(0,130)

#plt.legend(loc=1)
fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/sam_n_v_phi.pdf', transparent=True)
fig.show()

# dN/dphi v phi
fig, ax = plt.subplots(figsize=(8.325,7))

ch3_dat = np.loadtxt('CH3_Disks/var_n_v_phi.dat')
oh_dat = np.loadtxt('OH_Disks/var_n_v_phi.dat')

ax.plot(ch3_dat[:,0], ch3_dat[:,1], 'r-', label=r'$\rm{CH}_3$', linewidth=6)
plt_errorbars(ch3_dat[:,0], ch3_dat[:,1], ch3_dat[:,2])
ax.plot(oh_dat[:,0], oh_dat[:,1], 'b-', label=r'$\rm{OH}$', linewidth=6)
plt_errorbars(oh_dat[:,0], oh_dat[:,1], oh_dat[:,2])

ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$\chi_v$')

ax.set_xlim(0,2)
ax.set_xticks([0,1,2])
ax.set_ylim(0,222)

#ax.legend(loc=1)
fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/sam_sus_v_phi.pdf', transparent=True)
fig.show()

plt.close('all')

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8.5, 15))

ch3_dat = np.loadtxt('CH3_Disks/n_v_phi.dat')
oh_dat = np.loadtxt('OH_Disks/n_v_phi.dat')

ax1.plot(ch3_dat[:,0], ch3_dat[:,1], 'r-', label=r'$\rm{CH}_3$', linewidth=6)
plt_errorbars(ch3_dat[:,0], ch3_dat[:,1], ch3_dat[:,2], ax=ax1)
ax1.plot(oh_dat[:,0], oh_dat[:,1], 'b-', label=r'$\rm{OH}$', linewidth=6)
plt_errorbars(oh_dat[:,0], oh_dat[:,1], oh_dat[:,2], ax=ax1)


ch3_dat = np.loadtxt('CH3_Disks/var_n_v_phi.dat')
oh_dat = np.loadtxt('OH_Disks/var_n_v_phi.dat')

ax2.plot(ch3_dat[:,0], ch3_dat[:,1], 'r-', label=r'$\rm{CH}_3$', linewidth=6)
plt_errorbars(ch3_dat[:,0], ch3_dat[:,1], ch3_dat[:,2], ax=ax2)
ax2.plot(oh_dat[:,0], oh_dat[:,1], 'b-', label=r'$\rm{OH}$', linewidth=6)
plt_errorbars(oh_dat[:,0], oh_dat[:,1], oh_dat[:,2], ax=ax2)

ax2.set_xlim(0,2)
fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/sam_both_v_phi.pdf', transparent=True)
