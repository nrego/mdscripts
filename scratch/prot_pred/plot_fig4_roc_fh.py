from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc
from matplotlib import pyplot as plt 
import numpy as np
import os, glob

homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

from constants import k


'''
\documentclass[10pt]{article}
\usepackage[usenames]{color} %used for font color
\usepackage{amssymb} %maths
\usepackage{amsmath} %maths
\usepackage[utf8]{inputenc} %useful to type directly diacritic characters
\usepackage{xcolor}

\definecolor{dkorange}{RGB}{127,63,0}
\definecolor{pink}{RGB}{255,0,127}
\definecolor{gray}{RGB}{97,97,97}
\definecolor{dkpurple}{RGB}{63,0,127}
'''

beta = 1/(k * 300)
beta_phi_vals, avg_N, err_avg_N, chi, err_chi = [arr.squeeze() for arr in np.split(np.loadtxt('../prod/phi_sims/Nvphi.dat'), 5, 1)]
## Figure 4 d ##  
## Plot roc curve and susceptiblity w/ f_h

# Find values first values of chi that are < chi_max to left and right
#   to get phi_minus and phi_plus
chi_max_idx = np.argmax(chi)
chi_max = np.max(chi)
print('beta phi star: {}'.format(beta_phi_vals[chi_max_idx]))

chi_thresh_mask = chi < (0.5*chi_max)
chi_minus_idx = np.max(np.where(chi_thresh_mask[:chi_max_idx])) 
chi_plus_idx = np.min(np.where(chi_thresh_mask[chi_max_idx:])) + chi_max_idx 
beta_phi_minus = beta_phi_vals[chi_minus_idx]
beta_phi_plus = beta_phi_vals[chi_plus_idx]
print('beta phi -: {}   beta phi +: {}'.format(beta_phi_minus, beta_phi_plus))
#embed()
beta_phi, tp, fp, tn, fn, tpr, fpr, prec, f_h, f_1, mcc = [arr.squeeze() for arr in np.split(np.loadtxt('performance.dat'), 11, 1)]

best_idx = np.argmax(f_h)
minus_idx = np.argmin(np.abs((beta_phi)-beta_phi_minus))
plus_idx = np.argmin(np.abs((beta_phi)-beta_phi_plus))
star_idx = np.argmin(np.abs((beta_phi)-beta_phi_vals[chi_max_idx]))
### phi_opt (best d_h) ###
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.delete(fpr, [minus_idx, plus_idx, best_idx, star_idx]), np.delete(tpr, [minus_idx, plus_idx, best_idx, star_idx]), 'ko', markersize=16)
ax.plot(fpr[star_idx], tpr[star_idx], 'bD', markersize=20, label=r'$\phi^*$')
ax.plot(fpr[minus_idx], tpr[minus_idx], 'b<', markersize=20, label=r'$\phi_{-}$')
ax.plot(fpr[plus_idx], tpr[plus_idx], 'b>', markersize=20, label=r'$\phi_{+}$')
ax.plot(fpr[best_idx], tpr[best_idx], 'rX', markersize=30, label=r'$\phi_\mathrm{opt}$')
#ax.plot(fpr[indices], tpr[indices], 'bo', markersize=12)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])
ax.set_xticks([])
ax.set_yticks([])
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

ax2.plot(beta_phi, f_h, 'b-', linewidth=4)
ax2.set_ylabel(r'$d_h$', color='b')
ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()
fig.savefig('{}/Desktop/sus_dh_comp.pdf'.format(homedir), transparent=True)


## Just plot dh
fig, ax = plt.subplots(figsize=(5.5,5))
ax.plot(beta_phi, f_h, 'k-', linewidth=4)
ax.plot(np.delete(beta_phi, [minus_idx, best_idx, star_idx, plus_idx]), np.delete(f_h, [minus_idx, best_idx, plus_idx, star_idx]), 'ko', markersize=12)
ax.plot(beta_phi[star_idx], f_h[star_idx], 'bD', markersize=18)
ax.plot(beta_phi[best_idx], f_h[best_idx], 'rX', markersize=20)
ax.plot(beta_phi[minus_idx], f_h[minus_idx], 'b<', markersize=18)
ax.plot(beta_phi[plus_idx], f_h[plus_idx], 'b>', markersize=18)
ax.set_ylabel(r'$d_\mathrm{h}$')
ax.set_xlabel(r'$\beta \phi$')
ax.set_xlim(0,4)
fig.tight_layout()
fig.savefig('{}/Desktop/dh.pdf'.format(homedir), transparent=True)

print("beta phi opt: {}".format(beta_phi[best_idx]))

