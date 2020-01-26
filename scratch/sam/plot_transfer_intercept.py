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


### PLOT Transferability of Intercepts
#########################################
ds_bulk = np.load('sam_pattern_bulk_pure.npz')
ds_pure = np.load('sam_pattern_pure.npz')


# Skip pure hydroxyl or methyl
slc = slice(0, None, 2)
energies_pure = ds_pure['energies'][slc]
emin_pure = ds_pure['base_energy']
p_q = ds_pure['pq'][slc].astype(float)
errs_pure = ds_pure['err_energies'][slc]
dx = ds_pure['dx'][slc]
dy = ds_pure['dy'][slc]
dz = ds_pure['dz'][slc]

sa = dy*dz #+ 2*dx*(dy+dz)
feat_subvol = np.vstack((dy*dz, dy, dz)).T

energies_bulk = ds_bulk['energies']
errs_bulk = ds_bulk['err_energies']
emin_bulk = ds_bulk['base_energy']

assert np.array_equal(p_q, ds_bulk['pq'])
assert np.array_equal(dx, ds_bulk['dx'])
assert np.array_equal(dy, ds_bulk['dy'])
assert np.array_equal(dz, ds_bulk['dz'])


diffs = energies_pure - energies_bulk

## Fit regressions of dy, dz on P and Q
reg_p = linear_model.LinearRegression().fit(p_q[:,0].reshape(-1,1), dy)
reg_q = linear_model.LinearRegression().fit(p_q[:,1].reshape(-1,1), dz)
y0 = reg_p.intercept_
z0 = reg_q.intercept_

ly = reg_p.predict(p_q[:,0].reshape(-1,1))
lz = reg_q.predict(p_q[:,1].reshape(-1,1))

dp = reg_p.coef_[0]
dq = reg_q.coef_[0]

feat_pq = np.zeros_like(feat_subvol)
feat_pq[:,0] = p_q.prod(axis=1)
feat_pq[:,1:] = p_q

perf_mse_subvol, err_subvol, xvals, fit, reg_subvol = fit_leave_one(feat_subvol, energies_pure, fit_intercept=False)
a1, a2, a3 = reg_subvol.coef_
perf_mse_pq, err_pq, xvals, fit, reg_pq = fit_leave_one(feat_pq, energies_pure-emin_pure, fit_intercept=False)

reg_temp = linear_model.LinearRegression().fit(feat_pq, energies_pure)
reg_temp.intercept_ = a1*y0*z0 + a2*y0 + a3*z0
reg_temp.coef_[0] = a1*dp*dq
reg_temp.coef_[1] = (a1*z0 + a2)*dp
reg_temp.coef_[2] = (a1*y0 + a3)*dq

pred = reg_temp.predict(feat_pq)
err = energies_pure - pred
perf_mse = err**2
