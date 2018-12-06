from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc 
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from constants import k

def plt_errorbars(bb, vals, errs, **kwargs):
    ax = plt.gca()
    ax.fill_between(bb, vals-errs, vals+errs, alpha=0.5, facecolor='k', **kwargs)

homedir = os.environ['HOME']
savedir = '{}/Desktop'.format(homedir)

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

beta = 1 /(300*k)
## From ppi_analysis/[pdbid]/prod/phi_sims dir, after running whamerr and wham_n_v_phi.py ##

beta_phi_vals, avg_N, err_avg_N, chi, err_chi = [arr.squeeze() for arr in np.split(np.loadtxt('Nvphi.dat'), 5, 1)]
peak_sus_dat = np.loadtxt('peak_sus.dat')

# in case we don't have data at phi=0
start_pt = 0

start_idx = np.argmax(beta_phi_vals >= start_pt)
skip=1

myslice = slice(start_idx, None, skip)


## N v phi
fig, ax = plt.subplots(figsize=(5.5,5))

ax.errorbar(beta_phi_vals[myslice], avg_N[myslice], yerr=err_avg_N[myslice], fmt='k-o', linewidth=3)
ax.set_xlim(0, 4)
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)

ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$\langle N_v \rangle_\phi$')

fig.tight_layout()
fig.savefig('{}/n_v_phi.pdf'.format(savedir), transparent=True)
plt.close('all')


## chi v phi
fig, ax = plt.subplots(figsize=(5.45,5))

ax.errorbar(beta_phi_vals[myslice], chi[myslice], yerr=err_chi[myslice], fmt='k-o', linewidth=3)

ax.set_xlim(0, 4)
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)
#ax.vlines([peak_sus_dat[0]-peak_sus_dat[1], peak_sus_dat[0]+peak_sus_dat[1]], ymin=0, ymax=ymax)
ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$\chi_v$')

fig.tight_layout()
fig.savefig('{}/chi_v_phi.pdf'.format(savedir), transparent=True)
plt.close('all')




