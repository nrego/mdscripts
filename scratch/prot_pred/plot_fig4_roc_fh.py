from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc
from matplotlib import pyplot as plt 
import numpy as np
import os, glob

homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

from constants import k

beta = 1/(k * 300)
beta_phi_vals, avg_N, err_avg_N, chi, err_chi = [arr.squeeze() for arr in np.split(np.loadtxt('../prod/phi_sims/Nvphi.dat'), 5, 1)]
## Figure 4 d ##  
## Plot roc curve and susceptiblity w/ f_h

# Find values first values of chi that are < chi_max to left and right
#   to get phi_minus and phi_plus
chi_max_idx = np.argmax(chi)
chi_max = np.max(chi)
chi_thresh_mask = chi < (0.5*chi_max)
chi_minus_idx = np.max(np.where(chi_thresh_mask[:chi_max_idx])) 
chi_plus_idx = np.min(np.where(chi_thresh_mask[chi_max_idx:])) + chi_max_idx 
beta_phi_minus = beta_phi_vals[chi_minus_idx]
beta_phi_plus = beta_phi_vals[chi_plus_idx]
#embed()
phi, tp, fp, tn, fn, tpr, fpr, prec, f_h, f_1, mcc = [arr.squeeze() for arr in np.split(np.loadtxt('performance.dat'), 11, 1)]

best_idx = np.argmax(f_h)
minus_idx = np.argmin(np.abs((beta*phi)-beta_phi_minus))
plus_idx = np.argmin(np.abs((beta*phi)-beta_phi_plus))
### phi_opt (best d_h) ###
fig, ax = plt.subplots(figsize=(6,5.5))
ax.plot(np.delete(fpr, [minus_idx, plus_idx]), np.delete(tpr, [minus_idx, plus_idx]), 'ko', markersize=8)
ax.plot(fpr[best_idx], tpr[best_idx], 'bD', markersize=12, label=r'$\phi_\mathrm{opt}$')
ax.plot(fpr[minus_idx], tpr[minus_idx], 'b<', markersize=12, label=r'$\phi_{-}$')
ax.plot(fpr[plus_idx], tpr[plus_idx], 'b>', markersize=12, label=r'$\phi_{+}$')
#ax.plot(fpr[indices], tpr[indices], 'bo', markersize=12)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
fig.tight_layout()
fig.savefig('{}/Desktop/roc.pdf'.format(homedir), transparent=True)
plt.close('all')


fig, ax1 = plt.subplots(figsize=(7,5))
ax2 = ax1.twinx()
ax1.errorbar(beta_phi_vals, chi, yerr=err_chi, fmt='r-', linewidth=4)
#ax1.plot(beta_phi_vals[[chi_minus_idx, chi_max_idx, chi_plus_idx]], chi[[chi_minus_idx, chi_max_idx, chi_plus_idx]], 'o', markersize=10)
ax1.set_ylabel(r'$\chi_v$', color='r')
ax1.set_xlabel(r'$\beta \phi$')
ax1.tick_params(axis='y', labelcolor='r')

ax2.plot(beta*phi, f_h, 'b-', linewidth=4)
ax2.set_ylabel(r'$d_h$', color='b')
ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()
fig.savefig('{}/Desktop/sus_dh_comp.pdf'.format(homedir), transparent=True)


print("phi opt (kJ/mol): {}".format(phi[best_idx]))

