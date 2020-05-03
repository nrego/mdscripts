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

from IPython import embed

plt.close('all')

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


### PLOT SIMPLE REG ON k_O for 6 x 6 ####
#########################################

energies, feat_vec, states = extract_from_ds("data/sam_pattern_06_06.npz")

k_o = feat_vec[:,0]

print('\nExtracting sam data...')
p = q = 6


print('  ...Done\n')


n_dat = energies.size
indices = np.arange(n_dat)

perf_mse, err, xvals, fit, reg = fit_leave_one(k_o, energies, fit_intercept=True)
boot_int, boot_coef = fit_bootstrap(k_o, energies, fit_intercept=True)

poly_feat = np.vstack((k_o**2, k_o)).T
perf_mse_poly, err_poly, xvals2, fit_poly, reg_poly = fit_leave_one(poly_feat, energies, fit_intercept=True)


coef_err = boot_coef.std(axis=0, ddof=1).item()
inter_err = boot_int.std(axis=0, ddof=1)

print('M1 Regression:')
print('##############\n')
print('inter: {:.2f} ({:1.2e})  coef: {:.2f} ({:1.2e})'.format(reg.intercept_, inter_err, reg.coef_[0], coef_err))

plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
norm = plt.Normalize(-15,15)
err_kwargs = {"lw":.5, "zorder":0, "color":'k'}

#sc = ax.scatter(k_o, energies, s=50, c=norm(err), cmap='coolwarm_r', zorder=100)
sc = ax.scatter(k_o, energies, s=50, zorder=100, color='k', label=r'$f$')
#ax.errorbar(k_o, energies, errs, fmt='none', **err_kwargs)
ax.plot(xvals, fit, '-', linewidth=4, zorder=200, label='linear')    

ax.plot(xvals, fit_poly, '-', linewidth=4, zorder=300, label='quadratic')

ax.set_xticks([0,12,24,36])

fig.tight_layout()
fig.savefig('{}/Desktop/m1.pdf'.format(homedir), transparent=True)
ax.set_ylim(-100,-50)
plt.legend()
fig.savefig('{}/Desktop/m1_label.pdf'.format(homedir), transparent=True)


#### PLOT ERRORS VS KO ####

plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()

sc = ax.scatter(k_o, err)

ax.scatter(k_o, err_poly)
ax.set_xticks([0,12,24,36])
fig.savefig('{}/Desktop/m1_err.pdf'.format(homedir), transparent=True)


