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


### PLOT SIMPLE REG ON k_O for 6 x 6 ####
#########################################

ds = np.load('sam_pattern_06_06.dat.npz')

states = ds['states']
energies = ds['energies']

n_dat = energies.size
indices = np.arange(n_dat)

# Gen feat vec 
myfeat = np.zeros((energies.size, 1))

for i, state in enumerate(states):
    myfeat[i] = state.k_o


# Fit model - LOO CV

perf_mse, err, xvals, fit, reg = fit_leave_one(myfeat, energies)

fig = plt.figure(figsize=(7,6))
ax = fig.gca()

ax.plot(myfeat[:,0], energies, 'o', color='gray')
ax.plot(xvals, fit, 'k-', linewidth=4)
ax.set_xticks([0,12,24,36])

fig.tight_layout()
fig.savefig('{}/Desktop/m1.pdf'.format(homedir), transparent=True)



e_val = 212
diff = np.abs(energies - e_val)
diff_mask = diff < 1
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

k_val = 23
k_mask = myfeat[:,0] == k_val
pred_e = reg.predict(np.array([k_val]).reshape(1,-1)).item()
diff = energies[k_mask] - pred_e

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
ax.plot(pred, energies, 'ok')
ax.plot([130, 130], [290, 290], 'k-')
plt.savefig('{}/Desktop/fig_m1_parity.pdf'.format(homedir), transparent=True)


