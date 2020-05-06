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
from scratch.neural_net.lib import *

from IPython import embed

plt.close('all')

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


### PLOT SIMPLE REG ON k_O for 6 x 6 ####
#
#  Also, compare to best performing ANN
#########################################

n_hidden_layer = 3
n_node_hidden = 12

ds = np.load('data/sam_ann_ml_trials.npz')
all_nets = ds['all_nets']
all_perf_tot = ds['all_perf_tot']
all_perf_cv = ds['all_perf_cv']
all_n_params = ds['all_n_params']
trial_n_hidden_layer = ds['trial_n_hidden_layer']
trial_n_node_hidden = ds['trial_n_node_hidden']
n_sample = ds['n_sample'].item()

min_perf = all_perf_cv.min(axis=0)

i_hidden_layer = np.digitize(n_hidden_layer, trial_n_hidden_layer) - 1
i_node_hidden = np.digitize(n_node_hidden, trial_n_node_hidden) - 1

print("initializing ANN with n_hidden_layer: {:d}, n_node_hidden: {:d}".format(n_hidden_layer, n_node_hidden))

net = all_nets[i_hidden_layer, i_node_hidden]
#net_perf = all_perf_tot[i_hidden_layer, i_node_hidden]
net_perf = min_perf[i_hidden_layer, i_node_hidden]
net_n_params = all_n_params[i_hidden_layer, i_node_hidden]

print('\nExtracting sam data...')
feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('data/sam_pattern_06_06.npz', embed_pos_ext=False)
n_patch_dim = feat_vec.shape[1]

k_o = ols_feat[:,0]

p = q = 6

print('  ...Done\n')

assert energies.size == n_sample
indices = np.arange(n_sample)

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
fig = plt.figure(figsize=(6.5,6))
ax = fig.gca()
norm = plt.Normalize(-15,15)
err_kwargs = {"lw":.5, "zorder":0, "color":'k'}

#sc = ax.scatter(k_o, energies, s=50, c=norm(err), cmap='coolwarm_r', zorder=100)
sc = ax.scatter(k_o, energies, s=50, zorder=100, color='k', label=r'$f$')
#ax.errorbar(k_o, energies, errs, fmt='none', **err_kwargs)
ax.plot(xvals, fit, '-', linewidth=4, zorder=200, label=r'$\hat{f}$, linear')    

ax.plot(xvals, fit_poly, '-', linewidth=4, zorder=300, label=r'$\hat{f}$, quadratic')

ax.set_xticks([0,12,24,36])

fig.tight_layout()
fig.savefig('{}/Desktop/m1.pdf'.format(homedir), transparent=True)
ax.set_ylim(-100,-50)
plt.legend()
fig.savefig('{}/Desktop/m1_label.pdf'.format(homedir), transparent=True)


#### PLOT ERRORS VS KO ####

plt.close('all')
fig = plt.figure(figsize=(6.5,6))
ax = fig.gca()

sc = ax.scatter(k_o, err)

ax.scatter(k_o, err_poly)
ax.set_xticks([0,12,24,36])
fig.savefig('{}/Desktop/m1_err.pdf'.format(homedir), transparent=True)


## Now plot linear, quad, and ANN

labels = ['linear', 'quadratic', 'ANN']
n_params = np.array([2, 3, net_n_params])
mses = np.array([perf_mse.mean(), perf_mse_poly.mean(), net_perf])

aics = n_sample*np.log(mses) + 2*n_params
#aics -= aics.min()

def_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.close('all')
fig = plt.figure(figsize=(6,5))
ax = fig.gca()

ax.bar(np.arange(3), aics, color=[def_colors[0], def_colors[1], 'gray'], width=0.8)
ax.set_xticks([0,1,2])
ax.set_xticklabels([])

y_min, y_max = ax.get_ylim()
ax.set_ylim(2000,4200)
fig.tight_layout()

plt.savefig('{}/Desktop/bar_comparison'.format(homedir), transparent=True)

