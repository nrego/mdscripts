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

# If true, enforce endpoint constraints ('pure' patterns constrained to f_0)
do_constr = False
### PLOT SIMPLE REG ON k_O for 6 x 6 ####
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


# Fit model - LOO CV, constrained so that a pure hydroxyl group gives 
#   delta f = fo - fc
if do_constr:
    perf_mse, err, xvals, fit, reg = fit_leave_one_constr(myfeat, delta_f, eqcons=[constraint], args=args)
    boot_int, boot_coef = fit_bootstrap(myfeat, delta_f, fit_intercept=False)
    
else:
    perf_mse, err, xvals, fit, reg = fit_leave_one(myfeat, dg_bind, weights=np.ones_like(errs), fit_intercept=True)
    boot_int, boot_coef = fit_bootstrap(myfeat, dg_bind, fit_intercept=True)

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
if do_constr:
    ax.errorbar(myfeat[:,0], delta_f+f_c, fmt='o', color='gray', yerr=errs)
    ax.plot(xvals, fit+f_c, 'k-', linewidth=4)
else:
    sc = ax.scatter(myfeat[:,0], dg_bind, s=50, c=norm(err), cmap='coolwarm_r', zorder=100)
    ax.errorbar(myfeat[:,0], dg_bind, errs, fmt='none', **err_kwargs)
    ax.plot(xvals, fit, 'k-', linewidth=2, zorder=200)    

ax.set_xticks([0,12,24,36])

fig.tight_layout()
fig.savefig('{}/Desktop/m1.pdf'.format(homedir), transparent=True)


## Horizontal error slice (different k vals, but similar delta f)
##########################

e_val = -61
if do_constr:
    e_val = e_val - f_c
    diff = np.abs(delta_f - e_val)
else:
    diff = np.abs(dg_bind - e_val)

diff_mask = diff < 2
min_k = myfeat[diff_mask, 0].min()
max_k = myfeat[diff_mask, 0].max()
mid_k = np.round((e_val - reg.intercept_) / (reg.coef_[0]))
min_mask = diff_mask & (myfeat[:,0] == min_k)
max_mask = diff_mask & (myfeat[:,0] == max_k)
mid_mask = diff_mask & (myfeat[:,0] == mid_k)

min_idx = indices[min_mask][diff[min_mask].argmin()]
max_idx = indices[max_mask][diff[max_mask].argmin()]
mid_idx = indices[mid_mask][diff[mid_mask].argmin()]


min_state = states[min_idx]
max_state = states[max_idx]
mid_state = states[mid_idx]

plt.close('all')
min_state.plot()
plt.savefig('{}/Desktop/fig_min_k.pdf'.format(homedir), transparent=True)
plt.close('all')
max_state.plot()
plt.savefig('{}/Desktop/fig_max_k.pdf'.format(homedir), transparent=True)
plt.close('all')
mid_state.plot()
plt.savefig('{}/Desktop/fig_mid_k.pdf'.format(homedir), transparent=True)

plt.close('all')


### Vertical err - variety of delta_f's for patterns with same k_o
######################
k_val = 23
k_mask = myfeat[:,0] == k_val
pred_e = reg.predict(np.array([k_val]).reshape(1,-1)).item()

if do_constr:
    delta_f[k_mask] - pred_e
else:
    diff = dg_bind[k_mask] - pred_e

min_idx = indices[k_mask][diff.argmin()]
max_idx = indices[k_mask][diff.argmax()]
mid_idx = indices[k_mask][np.abs(diff).argmin()]

min_state = states[min_idx]
max_state = states[max_idx]
mid_state = states[mid_idx]

plt.close('all')
min_state.plot()
plt.savefig('{}/Desktop/fig_min_e.pdf'.format(homedir), transparent=True)
plt.close('all')
max_state.plot()
plt.savefig('{}/Desktop/fig_max_e.pdf'.format(homedir), transparent=True)
plt.close('all')
mid_state.plot()
plt.savefig('{}/Desktop/fig_mid_e.pdf'.format(homedir), transparent=True)

plt.close('all')

fig = plt.figure(figsize=(7,6))
pred = reg.predict(myfeat)
ax = fig.gca()
ax.plot(pred, delta_f, 'ok')
ax.plot([0, f_o-f_c], [0, f_o-f_c], 'k-', linewidth=4)
plt.savefig('{}/Desktop/fig_m1_parity.pdf'.format(homedir), transparent=True)


