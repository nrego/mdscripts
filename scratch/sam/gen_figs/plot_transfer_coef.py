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

do_constr = False

# k_o, n_oo, n_oe
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
err_bulk_06_06, err_bulk_04_09, err_bulk_04_04 = ds_bulk['err_energies'][-3:]

assert np.array_equal(ds_bulk['pq'][-3], np.array([6,6]))
assert np.array_equal(ds_bulk['pq'][-2], np.array([4,9]))
assert np.array_equal(ds_bulk['pq'][-1], np.array([4,4]))

energies_06_06 = ds_06_06['energies']
energies_04_04 = ds_04_04['energies']
energies_04_09 = ds_04_09['energies']

dg_bind_06_06 = energies_06_06 - e_bulk_06_06
dg_bind_04_09 = energies_04_09 - e_bulk_04_09
dg_bind_04_04 = energies_04_04 - e_bulk_04_04

err_06_06 = np.sqrt(ds_06_06['err_energies']**2 + err_bulk_06_06**2)
err_04_09 = np.sqrt(ds_04_09['err_energies']**2 + err_bulk_04_09**2)
err_04_04 = np.sqrt(ds_04_04['err_energies']**2 + err_bulk_04_04**2)

states_06_06 = ds_06_06['states']
states_04_04 = ds_04_04['states']
states_04_09 = ds_04_09['states']

feat_06_06 = extract_from_states(states_06_06)
feat_04_04 = extract_from_states(states_04_04)
feat_04_09 = extract_from_states(states_04_09)

print('\nExtracting pure models...')
reg_c = np.load('sam_reg_inter_c.npz')['reg'].item()
reg_o = np.load('sam_reg_inter_o.npz')['reg'].item()


print('  ...Done\n')

### Find coefs ###
###################
constraint = lambda alpha, X, y, x_o, f_c, f_o: np.dot(alpha, x_o) + (f_c - f_o)

### 6 x 6 ###
#############

p = q = 6

shape_arr = np.array([p*q, p, q])
f_c = f_c_06_06 = reg_c.predict(shape_arr.reshape(1,-1)).item()
f_o = f_o_06_06 = reg_o.predict(shape_arr.reshape(1,-1)).item()

delta_f = dg_bind_06_06 - f_c

# Get feature for pure hydroxyl of this shape (for second constraint)
state_pure = State(np.array([], dtype=int), ny=p, nz=q)
x_o = x_o_06_06 = np.array([state_pure.k_o, state_pure.n_oo, state_pure.n_oe])
args = (x_o, f_c, f_o)

if do_constr:
    perf_mse, err, xvals, fit, reg = fit_leave_one_constr(feat_06_06[:,indices], delta_f, eqcons=[constraint], args=args)
    boot_intercept, boot_coef = fit_bootstrap(feat_06_06[:,indices], delta_f, fit_intercept=False)
    assert np.allclose(reg.predict(x_o.reshape(1,-1)).item(), f_o - f_c)
else:
    perf_mse, err, xvals, fit, reg = fit_leave_one(feat_06_06[:,indices], dg_bind_06_06, fit_intercept=True)
    boot_intercept, boot_coef = fit_bootstrap(feat_06_06[:,indices], dg_bind_06_06, fit_intercept=True)
    block_intercept, block_coef = fit_block(feat_06_06[:,indices], dg_bind_06_06, fit_intercept=True)

print("\nDOING 6x6... (N={:02d})".format(energies_06_06.size))
print_data(reg, boot_intercept, boot_coef)

coef_06_06 = reg.coef_.copy()
err_coef_06_06 = np.array([boot_coef.std(axis=0, ddof=1), block_coef.std(axis=0, ddof=1)])

rsq = 1 - (perf_mse.mean() / dg_bind_06_06.var())

print("  perf: {:0.6f} (mse: {:.1f})".format(rsq, perf_mse.mean()))

### 4 x 9 ###
#############

p = 4
q = 9

shape_arr = np.array([p*q, p, q])
f_c = f_c_04_09 = reg_c.predict(shape_arr.reshape(1,-1)).item()
f_o = f_o_04_09 = reg_o.predict(shape_arr.reshape(1,-1)).item()

delta_f = dg_bind_04_09 - f_c

# Get feature for pure hydroxyl of this shape (for second constraint)
state_pure = State(np.array([], dtype=int), ny=p, nz=q)
x_o = x_o_04_09 = np.array([state_pure.k_o, state_pure.n_oo, state_pure.n_oe])
args = (x_o, f_c, f_o)

# perf_mse, err, xvals, fit, reg = fit_leave_one(feat_04_09[:,indices], delta_f, fit_intercept=False, weights=1/err_06_06)
if do_constr:
    perf_mse, err, xvals, fit, reg = fit_leave_one_constr(feat_04_09[:,indices], delta_f, eqcons=[constraint], args=args)
    boot_intercept, boot_coef = fit_bootstrap(feat_04_09[:,indices], delta_f, fit_intercept=False)
    assert np.allclose(reg.predict(x_o.reshape(1,-1)).item(), f_o - f_c)
else:
    perf_mse, err, xvals, fit, reg = fit_leave_one(feat_04_09[:,indices], dg_bind_04_09, fit_intercept=True)
    boot_intercept, boot_coef = fit_bootstrap(feat_04_09[:,indices], dg_bind_04_09, fit_intercept=True)    
    block_intercept, block_coef = fit_block(feat_04_09[:,indices], dg_bind_04_09, fit_intercept=True)

print("\nDOING 4x9... (N={:02d})".format(energies_04_09.size))
print_data(reg, boot_intercept, boot_coef)

coef_04_09 = reg.coef_.copy()
err_coef_04_09 = np.array([boot_coef.std(axis=0, ddof=1), block_coef.std(axis=0, ddof=1)])

rsq = 1 - (perf_mse.mean() / dg_bind_04_09.var())

print("  perf: {:0.6f} (mse: {:.1f})".format(rsq, perf_mse.mean()))


### 4 x 4 ###

p = 4
q = 4

shape_arr = np.array([p*q, p, q])
f_c = f_c_04_04 = reg_c.predict(shape_arr.reshape(1,-1)).item()
f_o = f_o_04_04 = reg_o.predict(shape_arr.reshape(1,-1)).item()

delta_f = dg_bind_04_04 - f_c

# Get feature for pure hydroxyl of this shape (for second constraint)
state_pure = State(np.array([], dtype=int), ny=p, nz=q)
x_o = x_o_04_04 = np.array([state_pure.k_o, state_pure.n_oo, state_pure.n_oe])
args = (x_o, f_c, f_o)

# perf_mse, err, xvals, fit, reg = fit_leave_one(feat_04_04[:,indices], delta_f, fit_intercept=False, weights=1/err_04_04)
if do_constr:
    perf_mse, err, xvals, fit, reg = fit_leave_one_constr(feat_04_04[:,indices], delta_f, eqcons=[constraint], args=args)
    boot_intercept, boot_coef = fit_bootstrap(feat_04_04[:,indices], delta_f, fit_intercept=False)
    assert np.allclose(reg.predict(x_o.reshape(1,-1)).item(), f_o - f_c)
else:
    perf_mse, err, xvals, fit, reg = fit_leave_one(feat_04_04[:,indices], dg_bind_04_04, fit_intercept=True)
    boot_intercept, boot_coef = fit_bootstrap(feat_04_04[:,indices], dg_bind_04_04, fit_intercept=True)    
    block_intercept, block_coef = fit_block(feat_04_04[:,indices], dg_bind_04_04, fit_intercept=True)

print("\nDOING 4x4... (N={:02d})".format(energies_04_04.size))
print_data(reg, boot_intercept, boot_coef)

coef_04_04 = reg.coef_.copy()
err_coef_04_04 = np.array([boot_coef.std(axis=0, ddof=1), block_coef.std(axis=0, ddof=1)])

rsq = 1 - (perf_mse.mean() / dg_bind_04_04.var())

print("  perf: {:0.6f} (mse: {:.1f})".format(rsq, perf_mse.mean()))


### ALL ###


def constraint_all(alpha, X, y, x_o_06_06, f_c_06_06, f_o_06_06,
                   x_o_04_09, f_c_04_09, f_o_04_09, x_o_04_04, f_c_04_04, f_o_04_04):

    c1 = np.dot(alpha, x_o_06_06) + (f_c_06_06 - f_o_06_06)
    c2 = np.dot(alpha, x_o_04_09) + (f_c_04_09 - f_o_04_09)
    c3 = np.dot(alpha, x_o_04_04) + (f_c_04_04 - f_o_04_04)

    #return np.array([c1, c2, c3])
    return [c1]

e_all = np.hstack((energies_06_06, energies_04_09, energies_04_04))
dg_bind_all = np.hstack((dg_bind_06_06, dg_bind_04_09, dg_bind_04_04))
delta_f_all = np.hstack((dg_bind_06_06-f_c_06_06, dg_bind_04_09-f_c_04_09, dg_bind_04_04-f_c_04_04))

feat_all = np.vstack((feat_06_06, feat_04_09, feat_04_04))
w_all = np.hstack((1/err_06_06, 1/err_04_09, 1/err_04_04))
states_all = np.concatenate((states_06_06, states_04_09, states_04_04))

args = (x_o_06_06, f_c_06_06, f_o_06_06, x_o_04_09, f_c_04_09, f_o_04_09, x_o_04_04, f_c_04_04, f_o_04_04)

#perf_mse, err, xvals, fit, reg = fit_leave_one(feat_all[:,indices], delta_f_all, fit_intercept=False)
perf_mse, err, xvals, fit, reg = fit_leave_one_constr(feat_all[:,indices], delta_f_all, f_eqcons=constraint_all, args=args)
boot_intercept, boot_coef = fit_bootstrap(feat_all[:,indices], delta_f_all, fit_intercept=False)
r_sq = 1 - (perf_mse.mean() / delta_f_all.var())

print("\nFINAL MODEL (N={:02d})".format(e_all.size))
print_data(reg, boot_intercept, boot_coef)
print("  Final performance: {:0.6f}".format(r_sq))
print("  (MSE: {:0.2f})".format(perf_mse.mean()))

np.save('sam_reg_coef', reg)
np.savez_compressed('sam_pattern_pooled', energies=e_all, dg_bind=dg_bind_all, delta_f=delta_f_all, weights=w_all, states=states_all, feat_vec=feat_all)

fig, ax = plt.subplots()
blah = np.arange(3)

ax.bar(blah, coef_06_06, yerr=err_coef_06_06[0])
ax.bar(blah, coef_06_06, yerr=err_coef_06_06[1])

ax.bar(blah, coef_04_09, yerr=err_coef_04_09[0], fmt='D', markersize=12)
ax.bar(blah, coef_04_09, yerr=err_coef_04_09[1], fmt='D', markersize=12, label='4 9')

ax.bar(blah, coef_04_04, yerr=err_coef_04_04[0], fmt='X', markersize=12)
ax.bar(blah, coef_04_04, yerr=err_coef_04_04[1], fmt='X', markersize=12, label='4 4')



