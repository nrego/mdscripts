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
ds = np.load('sam_pattern_pure.npz')

# Skip pure hydroxyl
slc = slice(5, None, 2)
energies = ds['energies'][slc]
feat_vec = ds['feat_vec'][slc].astype(float)
errs = ds['err_energies'][slc]

p = feat_vec[:,0]
q = feat_vec[:,1]

dx = ds['dx'][slc]
dy = ds['dy'][slc]
dz = ds['dz'][slc]

reg_dy = linear_model.LinearRegression()
reg_dz = linear_model.LinearRegression()

reg_dy.fit(p.reshape(-1,1), dy)
reg_dz.fit(q.reshape(-1,1), dz)

b1_y = (np.sqrt(3)/2)*0.5
b1_z = 0.5

int_y = (dy-b1_y*p).mean()
pseudo_dy = int_y + b1_y*p

int_z = (dz-b1_z*q).mean()
pseudo_dz = int_z + b1_z*q
#pseudo_dy = reg_dy.predict(p.reshape(-1,1))
#pseudo_dz = reg_dz.predict(q.reshape(-1,1))

reg_dy.fit(p.reshape(-1,1), dy)
reg_dz.fit(q.reshape(-1,1), dz)

vol = dx * dy * dz
sa = (dy*dz) + 2*dx*(dy+dz)

subvol_feat = np.vstack((dy*dz, dy+dz)).T

#feat_vec[:,1] = 0.5*np.sqrt(3) * feat_vec[:,1]
# Load in perturbation reg coefs
reg_coef = np.load('sam_reg_coef.npy').item()

# adjusts the hydroxyl patterns by removing the delta f
#e_adjusted = energies - np.dot(feat_vec[:,2:], reg_coef.coef_)
e_adjusted = energies

# PQ and (P+Q)
myfeat = np.zeros((feat_vec.shape[0], 2))

myfeat[:,0] = feat_vec[:,:2].prod(axis=1)
myfeat[:,1] = feat_vec[:,:2].sum(axis=1)

myfeat[:,0] = pseudo_dy*pseudo_dz
myfeat[:,1] = pseudo_dy+pseudo_dz

myfeat = np.zeros((feat_vec.shape[0], 3))
myfeat[:,0] = feat_vec[:,:2].prod(axis=1)
myfeat[:,1:] = feat_vec[:,:2]

# fit on just P*Q
perf_mse, err, xvals, fit, reg = fit_leave_one(myfeat[:,0].reshape(-1,1), e_adjusted, weights=1/errs)

perf_mse, err, xvals, fit, reg = fit_leave_one(myfeat, e_adjusted, weights=1/errs)
boot_intercept, boot_coef = fit_bootstrap(myfeat, e_adjusted, weights=1/errs)

#fig = plt.figure()
#ax = fig.gca(projection='3d')

pq_vals = np.arange(myfeat[:,0].min(), myfeat[:,0].max()+10, 10)
p_vals = np.arange(myfeat[:,1].min(), myfeat[:,1].max()+10, 10)

xx, yy = np.meshgrid(pq_vals, p_vals)
fn = lambda x, y: reg.intercept_ + reg.coef_[0]*x + reg.coef_[1]*y
vals = fn(xx, yy)

ax = plt.gca(projection='3d')
ax.plot_surface(xx, yy, vals, alpha=0.5)
ax.scatter(myfeat[:,0], myfeat[:,1], e_adjusted)
plt.close('all')


lim = np.abs(err).max() + 1

### Plot errs versus PQ ###

fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.scatter(myfeat[:,0], err)
ax.set_ylim(-lim, lim)
plt.savefig('{}/Desktop/err1.png'.format(homedir), transparent=True)

## Plot errs versus (P+Q)
plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.scatter(myfeat[:,1], err)
ax.set_ylim(-lim, lim)
plt.savefig('{}/Desktop/err2.png'.format(homedir), transparent=True)

## Plot errs versus just P
plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.scatter(feat_vec[:,0], err)
ax.set_ylim(-lim, lim)
plt.savefig('{}/Desktop/err_p.png'.format(homedir), transparent=True)

## Plot errs versus just Q
plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.scatter(feat_vec[:,0], err)
ax.set_ylim(-lim, lim)
plt.savefig('{}/Desktop/err_q.png'.format(homedir), transparent=True)



pred = reg.predict(myfeat)
plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.plot(np.array([0,320]), np.array([0,320]), 'k-', linewidth=4)
ax.scatter(pred, e_adjusted, color='orange', zorder=3)
ax.set_xticks([0, 100, 200, 300])
ax.set_yticks([0, 100, 200, 300])
plt.savefig('{}/Desktop/parity.png'.format(homedir), transparent=True)

np.save('sam_reg_inter', reg)



