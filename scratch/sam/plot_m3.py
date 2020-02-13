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

from scipy.optimize import fmin_slsqp

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
ds_bulk = np.load('sam_pattern_bulk_pure.npz')

print('\nExtracting sam data...')
p = q = 6
bulk_idx = np.where((ds_bulk['pq'] == (p,q)).all(axis=1))[0].item()
bulk_e = ds_bulk['energies'][bulk_idx]

print('  bulk energy: {:.2f}'.format(bulk_e))

shape_arr = np.array([p*q, p, q])

states = ds['states']
energies = ds['energies']
errs = ds['err_energies']

dg_bind = energies - bulk_e

# Sanity
for state in states:
    assert state.P == p and state.Q == q
print('  ...Done\n')

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
myfeat = np.zeros((energies.size, 3))

for i, state in enumerate(states):
    myfeat[i] = state.k_o, state.n_oo, state.n_oe


state_pure = State(np.array([],dtype=int), ny=p, nz=q)
x_o = np.array([state_pure.k_o, state_pure.n_oo, state_pure.n_oe])


# Fit model - LOO CV
constraint = lambda alpha, X, y, x_o, f_c, f_o: np.dot(alpha, x_o) + (f_c - f_o)
c1 = lambda alpha, X, y, x_o, f_c, f_o: np.dot(alpha, x_o[0]).item() + (f_c - f_o)
args = (x_o, f_c, f_o)

perf_mse, err_m1, xvals, fit, reg_m1 = fit_leave_one_constr(myfeat[:,0], delta_f, eqcons=[c1], args=args)

perf_mse, err, xvals, fit, reg = fit_leave_one_constr(myfeat, delta_f, eqcons=[constraint], args=args)

#perf_mse, err, xvals, fit, reg2 = fit_leave_one(myfeat, delta_f, fit_intercept=False)

k_vals = myfeat[:,0]

e_lim = np.ceil(np.abs(err_m1).max()) + 1
# Plot residuals v k_o
plt.close('all')

fig = plt.figure(figsize=(7,6.5))
ax = fig.gca()
ax.plot(k_vals, err_m1, 'o', label='Model 1')
ax.plot(k_vals, err, 'o', label='Model 3')
ax.set_xticks([0,12,24,36])
ax.set_ylim(-e_lim, e_lim)
ax.set_yticks([-20, -10, 0, 10, 20])
fig.tight_layout()
fig.savefig('{}/Desktop/res_comparison_ko.pdf'.format(homedir), transparent=True)

# Plot residuals v n_oo
plt.close('all')

fig = plt.figure(figsize=(7,6.5))
ax = fig.gca()
ax.plot(myfeat[:,1], err_m1, 'o', label='Model 1')
ax.plot(myfeat[:,1], err, 'o', label='Model 3')
ax.set_ylim(-e_lim, e_lim)
ax.set_yticks([-20, -10, 0, 10, 20])
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
ax.set_ylim(-e_lim, e_lim)
ax.set_yticks([-20, -10, 0, 10, 20])
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

