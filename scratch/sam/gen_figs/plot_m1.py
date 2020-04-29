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
sc = ax.scatter(k_o, energies, s=50, zorder=100, color='k')
#ax.errorbar(k_o, energies, errs, fmt='none', **err_kwargs)
ax.plot(xvals, fit, '-', linewidth=4, zorder=200)    

ax.plot(xvals, fit_poly, '-', linewidth=4, zorder=300)

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


