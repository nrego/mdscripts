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


### PLOT Transferability of Intercepts
#########################################
ds = np.load('sam_pattern_methyl.npz')

energies = ds['energies']
feat_vec = ds['feat_vec']
errs = ds['err_energies']

# fit on just P*Q
perf_mse, err, xvals, fit, reg = fit_leave_one(feat_vec[:,0].reshape(-1,1), energies, weights=1/errs)

myfeat = np.zeros((feat_vec.shape[0], 2))

myfeat[:,0] = feat_vec[:,1:].prod(axis=1)
myfeat[:,1] = feat_vec[:,1:].sum(axis=1)

perf_mse, err, xvals, fit, reg = fit_leave_one(myfeat, energies, weights=1/errs)
boot_intercept, boot_coef = fit_bootstrap(myfeat, energies, weights=1/errs)

#fig = plt.figure()
#ax = fig.gca(projection='3d')

pq_vals = np.arange(myfeat[:,0].min(), myfeat[:,0].max()+10, 10)
p_vals = np.arange(myfeat[:,1].min(), myfeat[:,1].max()+10, 10)

xx, yy = np.meshgrid(pq_vals, p_vals)
fn = lambda x, y: reg.intercept_ + reg.coef_[0]*x + reg.coef_[1]*y
vals = fn(xx, yy)

ax = plt.gca(projection='3d')
ax.plot_surface(xx, yy, vals, alpha=0.5)
ax.scatter(myfeat[:,0], myfeat[:,1], energies)

lim = np.abs(err).max() + 1

fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.scatter(myfeat[:,0], err)
ax.set_ylim(-lim, lim)
plt.savefig('{}/Desktop/err1.png'.format(homedir), transparent=True)

plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.scatter(myfeat[:,1], err)
ax.set_ylim(-lim, lim)
plt.savefig('{}/Desktop/err2.png'.format(homedir), transparent=True)

pred = reg.predict(myfeat)
plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.plot(np.array([0,320]), np.array([0,320]), 'k-', linewidth=4)
ax.scatter(pred, energies, color='orange', zorder=3)
ax.set_xticks([0, 100, 200, 300])
ax.set_yticks([0, 100, 200, 300])
plt.savefig('{}/Desktop/parity.png'.format(homedir), transparent=True)

np.save('sam_reg_inter', reg)



