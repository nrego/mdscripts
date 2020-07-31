
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


## 4 x 4 ##
energies_04_04, ols_feat_vec_04_04, states_04_04 = extract_from_ds('data/sam_pattern_04_04.npz')
err_energies_04_04 = np.load('data/sam_pattern_04_04.npz')['err_energies']
weights_04_04 = 1 / err_energies_04_04**2

feat_m2_04_04 = np.append(ols_feat_vec_04_04[:,0][:,None], ols_feat_vec_04_04[:,1:].sum(axis=1)[:,None], axis=1)


##################################


## Extract pure methyl patterns ##
##################################


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

##################################
##################################

## Plot expanded dataset ##

int_06_06, int_04_09, int_04_04 = reg.predict([[12,36],[13,36],[8,16]])

expand_feat_vec = np.vstack((ols_feat_vec_06_06, ols_feat_vec_04_09, ols_feat_vec_04_04))
#expand_energies = np.hstack((energies_06_06-int_06_06, energies_04_09-int_04_09, energies_04_04-int_04_04))
expand_energies = np.hstack((energies_06_06-energies_06_06.min(), energies_04_09-energies_04_09.min(), energies_04_04-energies_04_04.min()))

expand_perf_mse, expand_perf_wt_mse, expand_perf_r2, expand_err, expand_reg = fit_multi_k_fold(expand_feat_vec, expand_energies, fit_intercept=False)

boot_var_inter_expand, boot_var_coef_expand = boot_block_error(expand_feat_vec, expand_energies, fit_intercept=False)

##############################################################
### Plot Intercepts, coefs, for each model plus error bars ###
##############################################################

se_06_06 = np.append(np.sqrt(np.mean(boot_var_inter_06_06)), np.sqrt(np.mean(boot_var_coef_06_06, axis=0)))
se_expanded = np.append(np.sqrt(np.mean(boot_var_inter_expand)), np.sqrt(np.mean(boot_var_coef_expand, axis=0)))
ses = np.vstack((se_06_06, se_expanded))[:, 1:]

# now stack the coefs
coefs = np.vstack((reg_06_06.coef_, expand_reg.coef_))

n = 2
# indices for inter, coef1, coef2, coef3
indices = np.arange(n)
width = 1

colors = [(1,0,0,1), (1,1,0,1)]

plt.close('all')
fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12,3))

#ax1.bar(indices, coefs[:,0], yerr=ses[:,0], width=width, color=['r','y'])
ax2.bar(indices, coefs[:,1], yerr=ses[:,1], width=width, color=['r','y'])
ax3.bar(indices, coefs[:,2], yerr=ses[:,2], width=width, color=['r','y'])



ax2.set_ylim(-0.8, -0.5)
#ax2.set_yticks([])
#ax2.set_yticklabels([])
ax2.set_xticks([])

ax3.set_ylim(-1.15, -0.75)
#ax2.set_yticks([6, 6.5, 7.5])
#ax3.set_yticklabels([])
ax3.set_xticks([])

#fig.tight_layout()
plt.close('all')

fig, ax1 = plt.subplots()
ax1.bar(indices, coefs[:,0], yerr=ses[:,0], width=width, color=['r','y'])
ax1.set_ylim(6, 7.6)
ax1.set_yticks([6, 6.5, 7.0])
ax1.set_yticklabels([])
ax1.set_xticks([])
##################################
### Plot special 6x6 patterns ####
##################################

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
ax.bar(indices - width/2, energies[sort_idx], yerr=err_energies[sort_idx], label=r'$f$', width=width)
ax.bar(indices + width/2, pred[sort_idx], label=r'$\hat{f}$', width=width)
#ax.legend()
ax.set_xticks(indices)
ax.set_xticklabels([])
#ax.set_ylim(0, 400)

#ax.set_yticklabels([])
plt.savefig('{}/Desktop/fig_special'.format(homedir), transparent=True)

ax.set_ylim(215, 275)
ax.set_yticks([])
plt.savefig('{}/Desktop/fig_special_inset'.format(homedir), transparent=True)
plt.close('all')

## Plot 4x9 ###
for i in [205, 300, 100]:
    states_04_09[i].plot()
    plt.savefig('{}/Desktop/fig_04_09_{}'.format(homedir, i), transparent=True)
    plt.close('all')

## Plot 6 x 6 ##
for i in [320, 100, 250]:
    states_06_06[i].plot()
    plt.savefig('{}/Desktop/fig_06_06_{}'.format(homedir, i), transparent=True)
    plt.close('all')

# Plot 4 x 4 ##
for i in [140, 200, 70]:
    states_04_04[i].plot()
    plt.savefig('{}/Desktop/fig_04_04_{}'.format(homedir, i), transparent=True)
    plt.close('all')



plt.savefig('{}/Desktop/transfer_plot_comparison'.format(homedir), transparent=True)
plt.close('all')

