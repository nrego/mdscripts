

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

def_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_dashed_lines(x, ymin, ymax=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if ymax is None:
        ymax = - ymin

    ax.plot(x, np.ones_like(x)*ymin, **kwargs)
    ax.plot(x, np.ones_like(x)*ymax, **kwargs)

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
n_node_hidden = 24

ds = np.load('data/sam_ann_ml_trials.npz')
all_nets = ds['all_nets']
all_perf_tot = ds['all_perf_tot']
all_perf_cv = ds['all_perf_cv']
all_n_params = ds['all_n_params']
trial_n_hidden_layer = ds['trial_n_hidden_layer']
trial_n_node_hidden = ds['trial_n_node_hidden']
n_sample = ds['n_sample'].item()


i_hidden_layer = np.digitize(n_hidden_layer, trial_n_hidden_layer) - 1
i_node_hidden = np.digitize(n_node_hidden, trial_n_node_hidden) - 1

print("initializing ANN with n_hidden_layer: {:d}, n_node_hidden: {:d}".format(n_hidden_layer, n_node_hidden))

net = all_nets[i_hidden_layer, i_node_hidden]
#net_perf = all_perf_tot[i_hidden_layer, i_node_hidden]
net_perf = all_perf_cv[i_hidden_layer, i_node_hidden]
net_n_params = all_n_params[i_hidden_layer, i_node_hidden]


feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('data/sam_pattern_06_06.npz', embed_pos_ext=True)
aug_feat, aug_energies = hex_augment_data(feat_vec, energies, pos_ext, patch_indices)

dataset = SAMDataset(aug_feat, aug_energies)
net_pred = net(dataset.X).detach().numpy().squeeze()
net_err = aug_energies - net_pred

print('\nExtracting sam data...')
feat_vec, patch_indices, pos_ext, energies, ols_feat, states = load_and_prep('data/sam_pattern_06_06.npz', embed_pos_ext=False)
n_patch_dim = feat_vec.shape[1]
err_energies = np.load('data/sam_pattern_06_06.npz')['err_energies']

k_o = ols_feat[:,0]

p = q = 6

print('  ...Done\n')


assert energies.size == n_sample
indices = np.arange(n_sample)

perf_mse, perf_wt_mse, all_perf_r2, err, reg = fit_multi_k_fold(k_o.reshape(-1,1), energies, fit_intercept=True)

# Bootstrapped variance  
boot_inter, boot_coef = boot_block_error(k_o.reshape(-1,1), energies, fit_intercept=True)

inter_err = np.sqrt(boot_inter.mean())
coef_err = np.sqrt(boot_coef.mean())

poly_feat = np.vstack((k_o**2, k_o)).T
perf_mse_poly, perf_wt_mse_poly, all_perf_r2_poly, err_poly, reg_poly = fit_multi_k_fold(poly_feat, energies, fit_intercept=True)

xvals = np.arange(0, 37)
fit = reg.predict(xvals[:,None])
fit_poly = reg_poly.predict(np.vstack((xvals**2, xvals)).T)

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
#ax.plot(xvals, fit, '-', linewidth=4, zorder=200, label=r'$\hat{f}$, linear')    

#ax.plot(xvals, fit_poly, '-', linewidth=4, zorder=300, label=r'$\hat{f}$, quadratic')

ax.plot(xvals, fit, '-', linewidth=4, zorder=200, label=r'M1')    

ax.plot(xvals, fit_poly, '-', linewidth=4, zorder=300, label=r'M1/Q')


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

ax.scatter(k_o, net_err[::6], color='k')

#ax.scatter(k_o, net_err[::6], color='gray')
xmin, xmax = ax.get_xlim()
data_rmse = np.sqrt(np.mean(err_energies**2))
#ax.plot([-10, 46], [-data_rmse, -data_rmse], 'k--')
#ax.plot([-10, 46], [data_rmse, data_rmse], 'k--')
#ax.fill_between([-10, 46], -data_rmse, data_rmse, alpha=0.5)
plot_dashed_lines([-10, 46], -np.sqrt(np.mean(net_err[::6]**2)), color='k', linestyle='--', linewidth=2)
plot_dashed_lines([-10, 46], -np.sqrt(np.mean(err_poly**2)), color=def_colors[1], linestyle='--', linewidth=2)
plot_dashed_lines([-10, 46], -np.sqrt(np.mean(err**2)), color=def_colors[0], linestyle='--', linewidth=2)

ax.set_xticks([0,12,24,36])
ax.set_xlim(xmin, xmax)
ax.set_ylim(-25, 25)
fig.savefig('{}/Desktop/m1_err.pdf'.format(homedir), transparent=True)
plt.close('all')

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

ax.bar(np.arange(3), np.sqrt(mses), color=[def_colors[0], def_colors[1], 'k'], width=0.8)
ax.set_xticks([0,1,2])
ax.set_xticklabels([])
#ax.set_ylim(0, 7.5)

y_min, y_max = ax.get_ylim()
#ax.set_ylim(2000,4200)
fig.tight_layout()

plt.savefig('{}/Desktop/bar_comparison'.format(homedir), transparent=True)
plt.close('all')



