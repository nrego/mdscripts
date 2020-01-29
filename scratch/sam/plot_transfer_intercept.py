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

### PLOTTING SETUP ###
######################
plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


### Extract cavity creation FE's for bulk and near pure surfaces
#########################################
ds_bulk = np.load('sam_pattern_bulk_pure.npz')
ds_pure = np.load('sam_pattern_pure.npz')


# Skip pure hydroxyl or methyl
slc = slice(0, None, 2)
energies_pure = ds_pure['energies'][slc]
emin_pure = ds_pure['base_energy']
p_q = ds_pure['pq'][slc].astype(float)
errs_pure = ds_pure['err_energies'][slc]
dx = ds_pure['dx'][slc]
dy = ds_pure['dy'][slc]
dz = ds_pure['dz'][slc]

sa = dy*dz #+ 2*dx*(dy+dz)
feat_subvol = np.vstack((dy*dz, dy, dz)).T

energies_bulk = ds_bulk['energies']
errs_bulk = ds_bulk['err_energies']
emin_bulk = ds_bulk['base_energy']

assert np.array_equal(p_q, ds_bulk['pq'])
assert np.array_equal(dx, ds_bulk['dx'])
assert np.array_equal(dy, ds_bulk['dy'])
assert np.array_equal(dz, ds_bulk['dz'])

## Binding free energy to pure surface
diffs = energies_pure - energies_bulk
errs_diffs = np.sqrt(errs_bulk**2 + errs_pure**2)

## Regress Ly, Lz on P, Q ##
############################
perf_p, err_p, xvals_p, fit_p, reg_p = fit_leave_one(p_q[:,0].reshape(-1,1), dy)
perf_q, err_q, xvals_q, fit_q, reg_q = fit_leave_one(p_q[:,1].reshape(-1,1), dz)
y0 = reg_p.intercept_
z0 = reg_q.intercept_

ly = reg_p.predict(p_q[:,0].reshape(-1,1))
lz = reg_q.predict(p_q[:,1].reshape(-1,1))

dp = reg_p.coef_[0]
dq = reg_q.coef_[0]


fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.plot(xvals_p, fit_p)
ax.scatter(p_q[:,0], dy)
ax.plot(xvals_q, fit_q)
ax.scatter(p_q[:,1], dz)
fig.tight_layout()
#plt.show()

plt.close('all')

# PQ, P, Q
feat_pq = np.zeros_like(feat_subvol)
feat_pq[:,0] = p_q.prod(axis=1)
feat_pq[:,1:] = p_q

perf_mse, err_reg, xvals, fit, reg = fit_leave_one(feat_pq, diffs, fit_intercept=False)
boot_int, boot_coef = fit_bootstrap(feat_pq, diffs, fit_intercept=False)
lim_min = np.floor(diffs.min())
lim_max = np.ceil(diffs.max()) 

pred = reg.predict(feat_pq)

fig = plt.figure(figsize=(7,6))
ax = plt.gca()
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k-', linewidth=4)
ax.plot(pred, diffs, 'o', zorder=3, markersize=12, color='orange')
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)

fig.tight_layout()

ax.set_xticks([-300, -200, -100, 0])
plt.savefig('{}/Desktop/fig_inter_fit.png'.format(homedir), transparent=True)
plt.close('all')

np.save('sam_reg_inter', reg)

