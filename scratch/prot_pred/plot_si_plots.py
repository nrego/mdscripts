from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc 
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from constants import k

from IPython import embed

import scipy

## Plot Sus v phi, TPR,FPR, and PPV v phi, ROC, dh v phi, and f1 v phi
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
mpl.rcParams.update({'legend.fontsize':20})

beta = 1/(k * 300)
beta_phi_vals, avg_N, err_avg_N, smooth_chi, err_smooth_chi, chi, err_chi = [arr.squeeze() for arr in np.split(np.loadtxt('../phi_sims/NvPhi.dat'), 7, 1)]
## Figure 4 d ##  
## Plot roc curve and susceptiblity w/ f_h

# Find values first values of chi that are < chi_max to left and right
#   to get phi_minus and phi_plus
chi_max_idx = np.argmax(smooth_chi)
chi_max = np.max(smooth_chi)


chi_thresh_mask = smooth_chi < (0.5*chi_max)
chi_minus_idx = np.max(np.where(chi_thresh_mask[:chi_max_idx])) 
chi_plus_idx = np.min(np.where(chi_thresh_mask[chi_max_idx:])) + chi_max_idx 
beta_phi_minus = np.round(beta_phi_vals[chi_minus_idx], 2)
beta_phi_plus = beta_phi_vals[chi_plus_idx]

#embed()
beta_phi, tp, fp, tn, fn, tpr, fpr, prec, f_h, f_1, mcc = [arr.squeeze() for arr in np.split(np.loadtxt('performance.dat'), 11, 1)]

prec[(tp+fp)==0] = np.nan


## Find beta phi_opt and beta phi opt' ##
opt_idx = np.argmax(f_h)
opt_idx_prime = np.argmax(f_1)
minus_idx = np.where(beta_phi==beta_phi_minus)[0].item()
plus_idx = np.where(beta_phi==beta_phi_plus)[0].item()
star_idx = np.where(beta_phi==beta_phi_vals[chi_max_idx])[0].item()

print('beta phi *: {}'.format(beta_phi[star_idx]))
print('beta phi opt: {}   beta phi opt_prime: {}'.format(beta_phi[opt_idx], beta_phi[opt_idx_prime]))
print('beta phi -: {}   beta phi +: {}'.format(beta_phi[minus_idx], beta_phi[plus_idx]))


### SMOOTH CHI V PHI ###
fig, ax = plt.subplots(figsize=(5.45,5))

plot_errorbar(beta_phi, smooth_chi, err_smooth_chi, ax=ax, color='k')

#ax.plot(np.delete(beta_phi_vals[myslice], [chi_max_idx, chi_minus_idx, chi_plus_idx]), np.delete(chi[myslice], [chi_max_idx, chi_minus_idx, chi_plus_idx]), 'ko')
ax.plot(beta_phi[star_idx], smooth_chi[star_idx], 'bD', markersize=12, zorder=3)
ax.plot(beta_phi[minus_idx], smooth_chi[minus_idx], 'b<', markersize=12, zorder=3)
ax.plot(beta_phi[plus_idx], smooth_chi[plus_idx], 'b>', markersize=12, zorder=3)
ax.plot(beta_phi[opt_idx_prime], smooth_chi[opt_idx_prime], 'gP', markersize=12, zorder=3)
ax.plot(beta_phi[opt_idx], smooth_chi[opt_idx], 'rX', markersize=12, zorder=3)

ax.set_xlim(0, 4)
ymin, ymax = ax.get_ylim()
ax.set_ylim(0, ymax)
#ax.vlines([peak_sus_dat[0]-peak_sus_dat[1], peak_sus_dat[0]+peak_sus_dat[1]], ymin=0, ymax=ymax)
ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'$\chi_v$')

fig.tight_layout()
fig.savefig('{}/chi_v_phi.pdf'.format(savedir), transparent=True)
plt.close('all')

### END SMOOTH CHI V PHI ####



### PLOT ROC ###

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.delete(fpr, [minus_idx, plus_idx, opt_idx, opt_idx_prime, star_idx]), np.delete(tpr, [minus_idx, plus_idx, opt_idx, opt_idx_prime, star_idx]), 'ko', markersize=16)
ax.plot(fpr[star_idx], tpr[star_idx], 'bD', markersize=20, label=r'$\phi^*$')
ax.plot(fpr[minus_idx], tpr[minus_idx], 'b<', markersize=20, label=r'$\phi^{(-)}$')
ax.plot(fpr[plus_idx], tpr[plus_idx], 'b>', markersize=20, label=r'$\phi^{(+)}$')
ax.plot(fpr[opt_idx_prime], tpr[opt_idx_prime], 'gP', markersize=30, label=r"$\phi^\mathrm{opt'}$")
ax.plot(fpr[opt_idx], tpr[opt_idx], 'rX', markersize=30, label=r'$\phi^\mathrm{opt}$')
#ax.plot(fpr[indices], tpr[indices], 'bo', markersize=12)
ax.set_xlim(-0.02,1)
ax.set_ylim(0,1)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
#ax.set_xticks([])
#ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.set_xlabel('FPR')
#ax.set_ylabel('TPR')
fig.tight_layout()
#ax.set_xlim(10,20)
#ax.set_ylim(10,20)
#ax.legend(loc=10)
fig.savefig('{}/Desktop/roc.pdf'.format(homedir), transparent=True)
plt.close('all')

################


#### PLOT TPR, FPR, and PPV v beta phi ####

tnr = 1 - fpr
ppv = prec
c1, c2, c3 = 'c', 'm', '#8a6212'
## Plot TPR, 1-FPR v beta phi ##
fig, ax = plt.subplots(figsize=(5.5,5.5))
ax.plot(beta_phi, tpr, '-', color=c1, linewidth=4, label='TPR')
ax.plot(beta_phi[star_idx], tpr[star_idx], 'bD', markersize=18)
ax.plot(beta_phi[opt_idx_prime], tpr[opt_idx_prime], 'gP', markersize=20)
ax.plot(beta_phi[opt_idx], tpr[opt_idx], 'rX', markersize=20)
ax.plot(beta_phi[minus_idx], tpr[minus_idx], 'b<', markersize=18)
ax.plot(beta_phi[plus_idx], tpr[plus_idx], 'b>', markersize=18)

ax.plot(beta_phi, fpr, '-', color=c2, linewidth=4, label='FPR')
ax.plot(beta_phi[star_idx], fpr[star_idx], 'bD', markersize=18)
ax.plot(beta_phi[opt_idx_prime], fpr[opt_idx_prime], 'gP', markersize=20)
ax.plot(beta_phi[opt_idx], fpr[opt_idx], 'rX', markersize=20)
ax.plot(beta_phi[minus_idx], fpr[minus_idx], 'b<', markersize=18)
ax.plot(beta_phi[plus_idx], fpr[plus_idx], 'b>', markersize=18)

ax.plot(beta_phi, ppv, '-', color=c3, linewidth=4, label='PPV')
ax.plot(beta_phi[star_idx], ppv[star_idx], 'bD', markersize=18)
ax.plot(beta_phi[opt_idx_prime], ppv[opt_idx_prime], 'gP', markersize=20)
ax.plot(beta_phi[opt_idx], ppv[opt_idx], 'rX', markersize=20)
ax.plot(beta_phi[minus_idx], ppv[minus_idx], 'b<', markersize=18)
ax.plot(beta_phi[plus_idx], ppv[plus_idx], 'b>', markersize=18)

#ax.legend(handlelength=1)
#ax.set_ylim(3,4)
ax.set_yticks([])
#ax.set_xlabel(r'$\beta \phi$')
ax.set_xlim(0,4)
ax.set_ylim(0,1)
#ax.set_xticks([0,2,4])
ax.set_xticks([])
fig.tight_layout()
#ax.set_xlim(10,20)
#ax.set_ylim(10,20)
#ax.legend(loc=10)
fig.savefig('{}/Desktop/tpr_fpr.pdf'.format(homedir), transparent=True)
plt.close('all')
###############################################



##### d_h v beta phi #########
fig, ax = plt.subplots(figsize=(5.5,5.5))

ax.plot(beta_phi, f_h, 'r-', linewidth=4, label=r'$d_\mathrm{h}$')
#ax1.plot(beta_phi[star_idx], f_h[star_idx], 'bD', markersize=18)
#ax1.plot(beta_phi[opt_idx_prime], f_h[opt_idx_prime], 'gP', markersize=18)
ax.plot(beta_phi[opt_idx], f_h[opt_idx], 'rX', markersize=18)
#ax1.plot(beta_phi[minus_idx], f_h[minus_idx], 'b<', markersize=18)
#ax1.plot(beta_phi[plus_idx], f_h[plus_idx], 'b>', markersize=18)

ax.plot(beta_phi, f_1, 'g-', linewidth=4, label=r'$f_1$')
#ax2.plot(beta_phi[star_idx], f_1[star_idx], 'bD', markersize=18)
#ax2.plot(beta_phi[opt_idx], f_1[opt_idx], 'rX', markersize=18)
ax.plot(beta_phi[opt_idx_prime], f_1[opt_idx_prime], 'gP', markersize=18)
#ax2.plot(beta_phi[minus_idx], f_1[minus_idx], 'b<', markersize=18)
#ax2.plot(beta_phi[plus_idx], f_1[plus_idx], 'b>', markersize=18)
#ax.set_ylabel(r'$f_1$')

#ax.set_xlabel(r'$\beta \phi$')
ax.set_xlim(0,4)
ax.set_xticks([])
ax.set_ylim(0,1)
ax.set_yticks([])
#fig.legend(loc=10)
#ax.set_xlim(10,20)
#ax.set_ylim(10,20)
fig.tight_layout()
fig.savefig('{}/Desktop/dh.pdf'.format(homedir), transparent=True)
plt.close('all')

##############################
