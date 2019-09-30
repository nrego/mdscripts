from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc 
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from constants import k

from IPython import embed

def plot_errorbar(bb, dat, err, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(bb, dat, **kwargs)
    ax.fill_between(bb, dat-err, dat+err, alpha=0.5)

homedir = os.environ['HOME']
savedir = '{}/Desktop'.format(homedir)

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

beta = 1 /(300*k)
## From ppi_analysis/[pdbid]/prod/phi_sims dir, after running whamerr and wham_n_v_phi.py ##

beta_phi_vals, avg_N, err_avg_N, smooth_chi, err_smooth_chi, chi, err_chi = [arr.squeeze() for arr in np.split(np.loadtxt('NvPhi.dat'), 7, 1)]
peak_sus_dat = np.loadtxt('peak_sus.dat')

chi_max_idx = np.argmax(chi)
chi_max = np.max(chi)
chi_thresh_mask = chi < (0.5*chi_max)
chi_minus_idx = np.max(np.where(chi_thresh_mask[:chi_max_idx])) 
chi_plus_idx = np.min(np.where(chi_thresh_mask[chi_max_idx:])) + chi_max_idx 
beta_phi_minus = beta_phi_vals[chi_minus_idx]
beta_phi_plus = beta_phi_vals[chi_plus_idx]

# in case we don't have data at phi=0
start_pt = 0

start_idx = np.argmax(beta_phi_vals >= start_pt)
skip=5

myslice = slice(start_idx, None, skip)


## N v phi
fig, ax = plt.subplots(figsize=(5.5,5))

#ax.errorbar(beta_phi_vals[myslice], avg_N[myslice], yerr=err_avg_N[myslice], fmt='k-o', linewidth=3)
plot_errorbar(beta_phi_vals[myslice], avg_N[myslice], err_avg_N[myslice], ax=ax, color='k')
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

ax.errorbar(beta_phi_vals[myslice], chi[myslice], yerr=err_chi[myslice], fmt='k-', linewidth=3)
#ax.plot(beta_phi_vals, chi, 'k-', linewidth=3)
#ax.fill_between(beta_phi_vals, chi+err_chi, chi-err_chi, alpha=0.5)
ax.plot(np.delete(beta_phi_vals[myslice], [chi_max_idx, chi_minus_idx, chi_plus_idx]), np.delete(chi[myslice], [chi_max_idx, chi_minus_idx, chi_plus_idx]), 'ko')
ax.plot(beta_phi_vals[chi_max_idx], chi_max, 'bD', markersize=16, zorder=3)
ax.plot(beta_phi_minus, chi[chi_minus_idx], 'b<', markersize=16, zorder=3)
ax.plot(beta_phi_plus, chi[chi_plus_idx], 'b>', markersize=16, zorder=3)
ax.set_xlim(0, 4)
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)
#ax.vlines([peak_sus_dat[0]-peak_sus_dat[1], peak_sus_dat[0]+peak_sus_dat[1]], ymin=0, ymax=ymax)
ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$\chi_v$')

fig.tight_layout()
fig.savefig('{}/chi_v_phi.pdf'.format(savedir), transparent=True)
plt.close('all')


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,10))

ax1.errorbar(beta_phi_vals[myslice], avg_N[myslice], yerr=err_avg_N[myslice], fmt='k-o', linewidth=3)
ax2.errorbar(beta_phi_vals[myslice], chi[myslice], fmt='k-', yerr=err_chi[myslice], linewidth=3)
ax2.plot(np.delete(beta_phi_vals[myslice], [chi_max_idx, chi_minus_idx, chi_plus_idx]), np.delete(chi[myslice], [chi_max_idx, chi_minus_idx, chi_plus_idx]), 'ko')
ax2.plot(beta_phi_vals[chi_max_idx], chi_max, 'bD', markersize=16, zorder=3)
ax2.plot(beta_phi_minus, chi[chi_minus_idx], 'b<', markersize=16, zorder=3)
ax2.plot(beta_phi_plus, chi[chi_plus_idx], 'b>', markersize=16, zorder=3)
ax1.set_xticks([])
ax1.set_xlim(0,4)
ax2.set_xticks([0,2,4])
ax2.set_xlim(0,4)
ax2.set_xlabel(r'$\beta \phi$')

ax1.set_ylabel(r'$\langle N_v \rangle_\phi$')
ax2.set_ylabel(r'$\chi_v$')

fig.tight_layout()
fig.subplots_adjust(hspace=0.0)
fig.savefig('{}/both.pdf'.format(savedir), transparent=True)


print("beta phi_star: {}".format(beta_phi_vals[chi_max_idx]))
print("beta phi_minus: {}".format(beta_phi_minus))
print("beta phi_plus: {}".format(beta_phi_plus))


