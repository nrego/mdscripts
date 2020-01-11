from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *
plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


### PLOT Final model for 6x6 - on ko, noo, noe ####
#########################################

ds = np.load('sam_pattern_06_06.npz')

states = ds['states']
energies = ds['energies']
wt = 1 / ds['err_energies']

n_dat = energies.size
indices = np.arange(n_dat)

# Gen feat vec 
myfeat = np.zeros((energies.size, 3))

for i, state in enumerate(states):
    myfeat[i] = state.k_o, state.n_oo, state.n_oe


# Fit model - LOO CV
perf_mse, err_m1, xvals, fit, reg_m1 = fit_leave_one(myfeat[:,0].reshape(-1,1), energies, weights=wt)
perf_mse, err, xvals, fit, reg = fit_leave_one(myfeat, energies, weights=wt)
k_vals = myfeat[:,0]


# Plot residuals v k_o
plt.close('all')

fig = plt.figure(figsize=(7,6.5))
ax = fig.gca()
ax.plot(k_vals, err_m1, 'o', label='Model 1')
ax.plot(k_vals, err, 'o', label='Model 3')
ax.set_xticks([0,12,24,36])
#fig.legend()
#ax.set_ylim(100,110)
#plt.axis('off')
fig.tight_layout()
fig.savefig('{}/Desktop/res_comparison_ko.pdf'.format(homedir), transparent=True)

# Plot residuals v n_oo
plt.close('all')

fig = plt.figure(figsize=(7,6.5))
ax = fig.gca()
ax.plot(myfeat[:,1], err_m1, 'o', label='Model 1')
ax.plot(myfeat[:,1], err, 'o', label='Model 3')
#ax.set_xticks([0,12,24,36])
#fig.legend()
#ax.set_ylim(100,110)
#plt.axis('off')
fig.tight_layout()
fig.savefig('{}/Desktop/res_comparison_noo.pdf'.format(homedir), transparent=True)

# Plot residuals v n_oe
plt.close('all')

fig = plt.figure(figsize=(7,6.5))
ax = fig.gca()
ax.plot(myfeat[:,2], err_m1, 'o', label='Model 1')
ax.plot(myfeat[:,2], err, 'o', label='Model 3')
#ax.set_xticks([0,12,24,36])
#fig.legend()
#ax.set_ylim(100,110)
#plt.axis('off')
fig.tight_layout()
fig.savefig('{}/Desktop/res_comparison_noe.pdf'.format(homedir), transparent=True)



# Find errors of each model with k
sq_err_m1 = err_m1**2
sq_err = err**2

bin_assign = np.digitize(k_vals, bins=np.arange(38)) - 1
mse_with_k = np.zeros_like(np.arange(37))
mse_with_k_m1 = np.zeros_like(mse_with_k)

for i in range(37):
    mask = bin_assign == i
    mse_with_k[i] = sq_err[mask].mean()
    mse_with_k_m1[i] = sq_err_m1[mask].mean()

## Get reg coef errorbars from bootstrapping
boot_intercept, boot_coef = fit_bootstrap(myfeat, energies)

print("intercept: {:0.4f} ({:0.4f})".format(reg.intercept_, boot_intercept.std(ddof=1)))
print("coefs: {} ({})".format(reg.coef_, boot_coef.std(axis=0, ddof=1)))

