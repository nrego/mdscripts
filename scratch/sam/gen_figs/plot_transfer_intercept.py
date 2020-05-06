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

### PLOTTING SETUP ###
######################
plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


def construct_pure_model(ds_pure, ds_bulk, slc, suffix='', exclude=None):

    energies_pure = ds_pure['energies'][slc]
    emin_pure = ds_pure['base_energy'].item()
    p_q = ds_pure['pq'][slc].astype(float)
    errs_pure = ds_pure['err_energies'][slc]
    dx = ds_pure['dx'][slc]
    dy = ds_pure['dy'][slc]
    dz = ds_pure['dz'][slc]

    sa = 2*dy*dz + 2*dx*(dy+dz)

    feat_subvol = np.vstack((dy*dz, dy, dz)).T

    ## Bulk vols
    energies_bulk = ds_bulk['energies']
    errs_bulk = ds_bulk['err_energies']
    emin_bulk = ds_bulk['base_energy'].item()

    ## Sanity check that bulk and pure datasets line up as expected
    assert np.array_equal(p_q, ds_bulk['pq'])
    assert np.array_equal(dx, ds_bulk['dx'])
    assert np.array_equal(dy, ds_bulk['dy'])
    assert np.array_equal(dz, ds_bulk['dz'])

    assert energies_pure.size == energies_bulk.size


    # exclude data points, if desired
    if exclude is not None:
        energies_pure = np.delete(energies_pure, exclude)
        energies_bulk = np.delete(energies_bulk, exclude)
        errs_bulk = np.delete(errs_bulk, exclude)
        errs_pure = np.delete(errs_pure, exclude)
        p_q = np.delete(p_q, exclude, axis=0)

    ## Binding free energy to pure surface
    diffs = energies_pure - energies_bulk
    errs_diffs = np.sqrt(errs_bulk**2 + errs_pure**2)

    # Regress dg_bind on pq, p, and q
    ####################

    # pq, p, q
    feat_pq = np.zeros((energies_pure.size, 3))
    feat_pq[:,0] = p_q.prod(axis=1)
    feat_pq[:,1:] = p_q

    perf_mse, err_reg, xvals, fit, reg = fit_leave_one(feat_pq, diffs, fit_intercept=False)
    boot_int, boot_coef = fit_bootstrap(feat_pq, diffs, fit_intercept=False)
    coef_err = boot_coef.std(axis=0, ddof=1)

    ## Print out coefs and error bars
    print("\nmse: {:.2f}  r2: {:.4f}".format(perf_mse.mean(), 1 - (perf_mse.mean()/diffs.var())))
    print("  theta_pq: {:.2f}  theta_p: {:.2f}  theta_q: {:.2f}".format(*reg.coef_))
    print("           ({:.2f})         ({:.2f})         ({:.2f})".format(*coef_err))
    
    lim_min = np.floor(diffs.min())
    lim_max = np.ceil(diffs.max()) 

    pred = reg.predict(feat_pq)

    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k-', linewidth=4)
    s = ax.scatter(pred, diffs, zorder=3, s=200, cmap='bwr_r', norm=plt.Normalize(-6,6), c=err_reg, edgecolors='k')
    fig.colorbar(s, ticks=[-6,0,6])
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    lim_min = np.round(lim_min/100)*100
    lim_max = np.round(lim_max/100)*100
    xticks = np.arange(lim_min, lim_max+100, 100)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)

    fig.tight_layout()


    #ax.set_xticks([-300, -200, -100, 0])
    plt.savefig('{}/Desktop/fig_inter_fit{}.png'.format(homedir, suffix), transparent=True)
    #plt.show()
    plt.close('all')

    np.savez_compressed('sam_reg_inter{}'.format(suffix), reg=reg, coef_err=coef_err, coef=reg.coef_)

    ## Plot resolution
    dg = 0.05
    pvals = np.arange(0, p_q[:,0].max()+2, dg)
    qvals = np.arange(0, p_q[:,1].max()+2, dg)
    pp, qq = np.meshgrid(pvals, qvals, indexing='ij')
    ppqq = pp*qq

    fn = lambda p, q, pq, reg: reg.coef_[0]*pq + reg.coef_[1]*p + reg.coef_[2]*q

    vals = fn(pp, qq, ppqq, reg)

    #min_val = np.round(vals.min()/100)*100
    min_val = np.floor(diffs.min()/100)*100 
    max_val = np.ceil(diffs.max()/100)*100
    #max_val = np.round(vals.max()/100)*100
    e_vals = np.arange(min_val, max_val, 100)
    #ax = plt.gca(projection='3d')
    plt.close('all')
    fig = plt.figure(figsize=(6,8))
    ax = plt.gca()
    pc = ax.pcolormesh(pp, qq, vals, norm=plt.Normalize(min_val, max_val))
    
    c = ax.contour(pp, qq, vals, levels=e_vals, colors='k', linestyles='solid')
    fig.colorbar(pc, ticks=e_vals)

    ax.scatter(p_q[:,0], p_q[:,1], c=err_reg, norm=plt.Normalize(-6,6), cmap='bwr_r', edgecolors='k', s=200, zorder=3)

    fig.tight_layout()
    plt.savefig('{}/Desktop/fig_inter_2d{}.png'.format(homedir, suffix), transparent=True)
    #plt.show()

### Extract cavity creation FE's for bulk and near pure surfaces
#########################################
ds_bulk = np.load('sam_pattern_bulk_pure.npz')
ds_pure = np.load('sam_pattern_pure.npz')


# Pure surfaces

construct_pure_model(ds_pure, ds_bulk, slice(0, None, 2), suffix='_c')
construct_pure_model(ds_pure, ds_bulk, slice(1, None, 2), suffix='_o')



## Regress Ly, Lz on P, Q ##
############################
perf_p, err_p, xvals_p, fit_p, reg_p = fit_leave_one(p_q[:,0].reshape(-1,1), dy)
perf_q, err_q, xvals_q, fit_q, reg_q = fit_leave_one(p_q[:,1].reshape(-1,1), dz)
y0 = reg_p.intercept_
z0 = reg_q.intercept_

ly = reg_p.predict(p_q[:,0].reshape(-1,1))
lz = reg_q.predict(p_q[:,1].reshape(-1,1))

dp = reg_p.coef_[0]
dq = reg_q.coef_[0]


fig = plt.figure(figsize=(7,6))
ax = fig.gca()
ax.plot(xvals_p, fit_p, color='C5')
ax.scatter(p_q[:,0], dy, color='C5')
ax.plot(xvals_q, fit_q, 'C9')
ax.scatter(p_q[:,1], dz, color='C9')
fig.tight_layout()
#plt.show()

plt.close('all')