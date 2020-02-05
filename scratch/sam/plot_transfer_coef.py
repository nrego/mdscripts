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

indices = np.array([2,3,4])
def extract_from_states(states):
    feat_vec = np.zeros((states.size, 9))

    for i, state in enumerate(states):
        feat_vec[i] = state.P, state.Q, state.k_o, state.n_oo, state.n_oe, state.k_c, state.n_mm, state.n_me, state.n_mo


    return feat_vec

def print_data(reg, boot_intercept, boot_coef):
    print("    inter: {:0.2f} ({:0.4f})".format(reg.intercept_, boot_intercept.std(ddof=1)))
    errs = boot_coef.std(ddof=1, axis=0)

    print("    k_o: {:0.2f} ({:0.4f})".format(reg.coef_[0], errs[0]))
    print("    noo: {:0.2f} ({:0.4f})".format(reg.coef_[1], errs[1]))
    print("    noe: {:0.2f} ({:0.4f})".format(reg.coef_[2], errs[2]))


plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


### PLOT Transferability of coefs for 4x4, 6x6, 4x9 ####
#########################################

ds_06_06 = np.load('sam_pattern_06_06.npz')
ds_04_04 = np.load('sam_pattern_04_04.npz')
ds_04_09 = np.load('sam_pattern_04_09.npz')

ds_bulk = np.load('sam_pattern_bulk_pure.npz')
e_bulk_06_06, e_bulk_04_09, e_bulk_04_04 = ds_bulk['energies'][-3:]


energies_06_06 = ds_06_06['energies']
energies_04_04 = ds_04_04['energies']
energies_04_09 = ds_04_09['energies']

err_06_06 = ds_06_06['err_energies']
err_04_04 = ds_04_04['err_energies']
err_04_09 = ds_04_09['err_energies']
#err_06_06 = np.ones_like(err_06_06)
#err_04_04 = np.ones_like(err_04_04)
#err_04_09 = np.ones_like(err_04_09)

states_06_06 = ds_06_06['states']
states_04_04 = ds_04_04['states']
states_04_09 = ds_04_09['states']

feat_06_06 = extract_from_states(states_06_06)
feat_04_04 = extract_from_states(states_04_04)
feat_04_09 = extract_from_states(states_04_09)

### Find coefs ###

### 6 x 6 ###
perf_mse, err, xvals, fit, reg = fit_leave_one(feat_06_06[:,indices], energies_06_06-energies_06_06.min(), weights=1/err_06_06, fit_intercept=False)
boot_intercept, boot_coef = fit_bootstrap(feat_06_06[:,indices], energies_06_06-energies_06_06.min(), weights=1/err_06_06, fit_intercept=False)

print("\nDOING 6x6... (N={:02d})".format(energies_06_06.size))
print_data(reg, boot_intercept, boot_coef)
rsq = 1 - (perf_mse.mean() / energies_06_06.var())
print("  perf: {:0.6f}".format(rsq))

### 4 x 9 ###
perf_mse, err, xvals, fit, reg = fit_leave_one(feat_04_09[:,indices], energies_04_09-energies_04_09.min(), weights=1/err_04_09, fit_intercept=False)
boot_intercept, boot_coef = fit_bootstrap(feat_04_09[:,indices], energies_04_09-energies_04_09.min(), weights=1/err_04_09, fit_intercept=False)

print("\nDOING 4x9... (N={:02d})".format(energies_04_09.size))
print_data(reg, boot_intercept, boot_coef)
rsq = 1 - (perf_mse.mean() / energies_04_09.var())
print("  perf: {:0.6f}".format(rsq))

### 4 x 4 ###
perf_mse, err, xvals, fit, reg = fit_leave_one(feat_04_04[:,indices], energies_04_04-energies_04_04.min(), weights=1/err_04_04, fit_intercept=False)
boot_intercept, boot_coef = fit_bootstrap(feat_04_04[:,indices], energies_04_04-energies_04_04.min(), weights=1/err_04_04, fit_intercept=False)

print("\nDOING 4x4... (N={:02d})".format(energies_04_04.size))
print_data(reg, boot_intercept, boot_coef)
rsq = 1 - (perf_mse.mean() / energies_04_04.var())
print("  perf: {:0.6f}".format(rsq))

e_all = np.hstack((energies_06_06-energies_06_06.min(), energies_04_09-energies_04_09.min(), energies_04_04-energies_04_04.min()))
e_all2 = np.hstack((energies_06_06, energies_04_09, energies_04_04))
dg_bind = np.hstack((energies_06_06-e_bulk_06_06, energies_04_09-e_bulk_04_09, energies_04_04-e_bulk_04_04))


feat_all = np.vstack((feat_06_06, feat_04_09, feat_04_04))
w_all = np.hstack((1/err_06_06, 1/err_04_09, 1/err_04_04))
states_all = np.concatenate((states_06_06, states_04_09, states_04_04))

perf_mse, err, xvals, fit, reg = fit_leave_one(feat_all[:,indices], e_all, weights=w_all, fit_intercept=False)
boot_intercept, boot_coef = fit_bootstrap(feat_all[:,indices], e_all, weights=w_all, fit_intercept=False)
r_sq = 1 - (perf_mse.mean() / e_all.var())

print("\nFINAL MODEL (N={:02d}".format(e_all.size))
print_data(reg, boot_intercept, boot_coef)
print("  Final performance: {:0.6f}".format(r_sq))
print("  (MSE: {:0.2f})".format(perf_mse.mean()))

np.save('sam_reg_coef', reg)
np.savez_compressed('sam_pattern_pooled', energies=e_all2, delta_e=e_all, dg_bind=dg_bind, weights=w_all, states=states_all, feat_vec=feat_all)

