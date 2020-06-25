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

import itertools

from sklearn import datasets, linear_model
from sklearn.cluster import AgglomerativeClustering



np.set_printoptions(precision=3)

# For SAM schematic pattern plotting
figsize = (10,10)

mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':34})

homedir = os.environ['HOME']

## 6 x 6 ##
energies_06_06, ols_feat_vec_06_06, states_06_06 = extract_from_ds('data/sam_pattern_06_06.npz')
err_energies_06_06 = np.load('data/sam_pattern_06_06.npz')['err_energies']
weights_06_06 = 1 / err_energies_06_06**2

feat_m2_06_06 = np.append(ols_feat_vec_06_06[:,0][:,None], ols_feat_vec_06_06[:,1:].sum(axis=1)[:,None], axis=1)

perf_mse_06_06, perf_wt_mse_06_06, perf_r2_06_06, err_06_06, reg_06_06 = fit_multi_k_fold(ols_feat_vec_06_06, energies_06_06, weights=weights_06_06, do_weighted=False)

state_06_06 = states_06_06[np.argwhere(ols_feat_vec_06_06[:,0] == 0).item()]

yprime_06_06 = energies_06_06 - reg_06_06.intercept_
reg2_06_06 = linear_model.LinearRegression(fit_intercept=False)
reg2_06_06.fit(ols_feat_vec_06_06, yprime_06_06)

## Bootstrap error bars on coefs
boot_var_inter_06_06, boot_var_coef_06_06 = boot_block_error(ols_feat_vec_06_06, energies_06_06)

## 4 x 9 ##
energies_04_09, ols_feat_vec_04_09, states_04_09 = extract_from_ds('data/sam_pattern_04_09.npz')
err_energies_04_09 = np.load('data/sam_pattern_04_09.npz')['err_energies']
weights_04_09 = 1 / err_energies_04_09**2

feat_m2_04_09 = np.append(ols_feat_vec_04_09[:,0][:,None], ols_feat_vec_04_09[:,1:].sum(axis=1)[:,None], axis=1)

perf_mse_04_09, perf_wt_mse_04_09, perf_r2_04_09, err_04_09, reg_04_09 = fit_multi_k_fold(ols_feat_vec_04_09, energies_04_09, weights=weights_04_09, do_weighted=False)

state_04_09 = states_04_09[np.argwhere(ols_feat_vec_04_09[:,0] == 0).item()]

yprime_04_09 = energies_04_09 - reg_04_09.intercept_
reg2_04_09 = linear_model.LinearRegression(fit_intercept=False)
reg2_04_09.fit(ols_feat_vec_04_09, yprime_04_09)

## Bootstrap error bars on coefs
boot_var_inter_04_09, boot_var_coef_04_09 = boot_block_error(ols_feat_vec_04_09, energies_04_09)


## 4 x 4 ##
energies_04_04, ols_feat_vec_04_04, states_04_04 = extract_from_ds('data/sam_pattern_04_04.npz')
err_energies_04_04 = np.load('data/sam_pattern_04_04.npz')['err_energies']
weights_04_04 = 1 / err_energies_04_04**2

feat_m2_04_04 = np.append(ols_feat_vec_04_04[:,0][:,None], ols_feat_vec_04_04[:,1:].sum(axis=1)[:,None], axis=1)

perf_mse_04_04, perf_wt_mse_04_04, perf_r2_04_04, err_04_04, reg_04_04 = fit_multi_k_fold(ols_feat_vec_04_04, energies_04_04, weights=weights_04_04, do_weighted=False)

state_04_04 = states_04_04[np.argwhere(ols_feat_vec_04_04[:,0] == 0).item()]

yprime_04_04 = energies_04_04 - reg_04_04.intercept_
reg2_04_04 = linear_model.LinearRegression(fit_intercept=False)
reg2_04_04.fit(ols_feat_vec_04_04, yprime_04_04)

## Bootstrap error bars on coefs
boot_var_inter_04_04, boot_var_coef_04_04 = boot_block_error(ols_feat_vec_04_04, energies_04_04)


# 3 x 3 ##
energies_03_03, ols_feat_vec_03_03, states_03_03 = extract_from_ds('data/sam_pattern_03_03.npz')
err_energies_03_03 = np.load('data/sam_pattern_03_03.npz')['err_energies']
weights_03_03 = 1 / err_energies_03_03**2

feat_m2_03_03 = np.append(ols_feat_vec_03_03[:,0][:,None], ols_feat_vec_03_03[:,1:].sum(axis=1)[:,None], axis=1)

perf_mse_03_03, perf_wt_mse_03_03, perf_r2_03_03, err_03_03, reg_03_03 = fit_multi_k_fold(ols_feat_vec_03_03, energies_03_03, weights=weights_03_03, do_weighted=False)

state_03_03 = states_03_03[np.argwhere(ols_feat_vec_03_03[:,0] == 0).item()]

yprime_03_03 = energies_03_03 - reg_03_03.intercept_
reg2_03_03 = linear_model.LinearRegression(fit_intercept=False)
reg2_03_03.fit(ols_feat_vec_03_03, yprime_03_03)

## Bootstrap error bars on coefs
boot_var_inter_03_03, boot_var_coef_03_03 = boot_block_error(ols_feat_vec_03_03, energies_03_03)



### Plot Intercepts, coefs, for each model plus error bars ###
##############################################################

se_06_06 = np.append(np.sqrt(np.mean(boot_var_inter_06_06)), np.sqrt(np.mean(boot_var_coef_06_06, axis=0)))
se_04_09 = np.append(np.sqrt(np.mean(boot_var_inter_04_09)), np.sqrt(np.mean(boot_var_coef_04_09, axis=0)))
se_04_04 = np.append(np.sqrt(np.mean(boot_var_inter_04_04)), np.sqrt(np.mean(boot_var_coef_04_04, axis=0)))
se_03_03 = np.append(np.sqrt(np.mean(boot_var_inter_03_03)), np.sqrt(np.mean(boot_var_coef_03_03, axis=0)))

ses = np.vstack((se_06_06, se_04_09, se_04_04))

# now stack the coefs
coefs = np.vstack((reg_06_06.coef_, reg_04_09.coef_, reg_04_04.coef_))
ints = np.array([reg_06_06.intercept_, reg_04_09.intercept_, reg_04_04.intercept_])

reg_params = np.hstack((ints[:,None], coefs))

n = 3
# indices for inter, coef1, coef2, coef3
indices = np.arange(n)
width = 1

plt.close('all')
fig, axes = plt.subplots(1, 4, figsize=(12,3))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color'][:n]

y_min = [30, 6, -1.2, -1.6]
y_max = [140, 9.3, -0.5, -0.75]

for i, ax in enumerate(axes):
    ax.bar(indices, reg_params[:,i], yerr=ses[:,i], color=colors)
    ax.set_xticks([])
    ax.set_ylim(y_min[i], y_max[i])

plt.subplots_adjust(wspace=1)
plt.tight_layout()

plt.savefig('{}/Desktop/transfer_plot_comparison'.format(homedir), transparent=True)
plt.close('all')

## Extract pure methyl patterns ##


base_energy = 2.442375429221714356e+00
ds = np.load('data/pure_nonpolar.npz')
pure_energies = ds['energies']
p = ds['p']
q = ds['q']
pq = p*q

pure_feat_vec = np.vstack(((p+q),pq)).T

all_perf_mse, all_perf_wt_mse, all_perf_r2, err, reg = fit_multi_k_fold(pure_feat_vec, pure_energies-base_energy, k=17, fit_intercept=False)

reg.intercept_ = base_energy
pred = reg.predict(pure_feat_vec)

plt.close('all')
fig, ax = plt.subplots(figsize=(6,6))
ax.plot([0,312],[0,312], 'k-', linewidth=4)
ax.plot(pure_energies, pred, 'mX', markersize=20)
ax.set_xticks([0,100,200,300])
ax.set_yticks([0,100,200,300])

plt.savefig("{}/Desktop/parity_pure".format(homedir), transparent=True)
plt.close('all')


### Plot special patterns ####
ds = np.load('data/sam_special_patterns.dat.npz')
states = ds['states']
names = ds['names']
energies = ds['energies']
err_energies = ds['err_energies']
feat_vec = ds['feat_vec']

pred = reg_06_06.predict(feat_vec)

for name, state, this_e, this_pred in zip(names, states, energies, pred):
    plt.close('all')
    state.plot()
    plt.savefig('{}/Desktop/{}'.format(homedir, name), transparent=True)
    plt.close('all')

    print("{}. act f: {:.2f}.  pred f: {:.2f}".format(name, this_e, this_pred))


sort_idx = np.argsort(energies)
plt.close('all')
indices = np.arange(9)
fig, ax = plt.subplots(figsize=(12,6))
width = 0.35
ax.bar(indices - width/2, energies[sort_idx], label=r'$f$', width=width)
ax.bar(indices + width/2, pred[sort_idx], label=r'$\hat{f}$', width=width)
#ax.legend()
ax.set_xticks(indices)
ax.set_xticklabels([])
ax.set_ylim(0, 300)

#ax.set_yticklabels([])
plt.savefig('{}/Desktop/fig_special'.format(homedir), transparent=True)
plt.close('all')

## Plot 4x9 ###
states_04_09[205].plot()
plt.savefig('{}/Desktop/fig_04_09'.format(homedir), transparent=True)
plt.close('all')

## Plot 6 x 6 ##
states_06_06[320].plot()
plt.savefig('{}/Desktop/fig_06_06'.format(homedir), transparent=True)
plt.close('all')

# Plot 4 x 4 ##
states_04_04[140].plot()
plt.savefig('{}/Desktop/fig_04_04'.format(homedir), transparent=True)
plt.close('all')




