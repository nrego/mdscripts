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
mpl.rcParams.update({'legend.fontsize':18})

### PLOT SIMPLE REG ON k_O for 6 x 6 ####
### Comparison of no constr linear, constrained ends, and poly fit
#########################################

ds = np.load('sam_pattern_06_06.npz')
ds_bulk = np.load('sam_pattern_bulk_pure.npz')


print('\nExtracting sam data...')
p = q = 6
bulk_idx = np.where((ds_bulk['pq'] == (p,q)).all(axis=1))[0].item()
bulk_e = ds_bulk['energies'][bulk_idx]

print('  bulk energy: {:.2f}'.format(bulk_e))

states = ds['states']
energies = ds['energies']
errs = ds['err_energies']

dg_bind = energies - bulk_e

# Sanity
for state in states:
    assert state.P == p and state.Q == q
print('  ...Done\n')

shape_arr = np.array([p*q, p, q])

print('\nExtracting pure models...')
reg_c = np.load('sam_reg_inter_c.npz')['reg'].item()
reg_o = np.load('sam_reg_inter_o.npz')['reg'].item()

f_c = reg_c.predict(shape_arr.reshape(1,-1)).item()
f_o = reg_o.predict(shape_arr.reshape(1,-1)).item()


print('  ...Done\n')

delta_f = dg_bind - f_c

n_dat = delta_f.size
indices = np.arange(n_dat)

# Gen feat vec 
myfeat = np.zeros((energies.size, 1))

for i, state in enumerate(states):
    myfeat[i] = state.k_o

constraint = lambda alpha, X, y, f_c, f_o: (alpha*36).item() + (f_c - f_o)
args = (f_c, f_o)


# Constrained at ends, linear fit
perf_mse_constr, err_constr, xvals, fit_constr, reg_constr = fit_leave_one_constr(myfeat, delta_f, eqcons=[constraint], args=args)
boot_int, boot_coef = fit_bootstrap(myfeat, delta_f, fit_intercept=False)

# Linear fit, no endpoint constraints
perf_mse, err, xvals, fit, reg = fit_leave_one(myfeat, dg_bind, weights=np.ones_like(errs), fit_intercept=True)
boot_int, boot_coef = fit_bootstrap(myfeat, dg_bind, fit_intercept=True)

# 2nd degree poly fit
poly_feat = np.hstack((myfeat**2, myfeat))
perf_mse_poly, err_poly, xvals2, fit_poly, reg_poly = fit_leave_one(poly_feat, dg_bind, weights=np.ones_like(errs), fit_intercept=True)

plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
norm = plt.Normalize(-15,15)

sc = ax.scatter(myfeat[:,0], dg_bind, s=50, color='k')
ax.plot(xvals, fit, label='m1, linear', linewidth=4)
#ax.plot(xvals, fit_constr+f_c, label='m1, constrained', linewidth=4)
ax.plot(xvals, fit_poly, label='m1, poly', linewidth=4)
ax.set_xticks([0,12,24,36])
#ax.set_ylim(200,300)
#fig.legend(loc=10)
#plt.axis('off')
fig.tight_layout()
fig.savefig('{}/Desktop/m1_comp.pdf'.format(homedir), transparent=True)

### Now for residuals..
plt.close('all')
fig = plt.figure(figsize=(7,6))
ax = fig.gca()
norm = plt.Normalize(-15,15)

#sc = ax.scatter(myfeat[:,0], dg_bind, s=50, color='k')
ax.scatter(myfeat[:,0], err, label='m1, linear', s=50)
#ax.scatter(myfeat[:,0], err_constr, label='m1, constrained', s=50)
ax.scatter(myfeat[:,0], err_poly, label='m1, poly', s=50)
ax.set_xticks([0,12,24,36])
#ax.set_ylim(200,300)
#fig.legend(loc=10)
#plt.axis('off')
fig.tight_layout()
fig.savefig('{}/Desktop/m1_err_comp.pdf'.format(homedir), transparent=True)

