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

# Skip pure hydroxyl
slc = slice(1, None, 2)
energies_pure = ds_pure['energies'][slc]
emin_pure = ds_pure['base_energy']
p_q = ds_pure['pq'][slc].astype(float)
errs_pure = ds_pure['err_energies'][slc]
dx = ds_pure['dx'][slc]
dy = ds_pure['dy'][slc]
dz = ds_pure['dz'][slc]

sa = dy*dz #+ 2*dx*(dy+dz)
feat = np.vstack((dy*dz, dy, dz)).T

energies_bulk = ds_bulk['energies']
errs_bulk = ds_bulk['err_energies']
emin_bulk = ds_bulk['base_energy']

assert np.array_equal(p_q, ds_bulk['pq'])
assert np.array_equal(dx, ds_bulk['dx'])
assert np.array_equal(dy, ds_bulk['dy'])
assert np.array_equal(dz, ds_bulk['dz'])


#diffs = energies_pure - energies_bulk
#plt.errorbar(sa, energies_bulk, yerr=errs_bulk, fmt='o')
#plt.show()

perf_mse_bulk, err_bulk, xvals, fit, reg_bulk = fit_leave_one(feat, energies_bulk, fit_intercept=False)
alpha1, alpha2, alpha3 = reg_bulk.coef_ 
plt.plot(xvals, fit)
plt.errorbar(sa, energies_bulk, yerr=errs_bulk, fmt='o')
plt.show()


## Relate P,Q to dy, dz
dq = 0.5
dp = 0.5*np.sqrt(3)*dq

y0 = np.mean(dy - dp*p_q[:,0])
z0 = np.mean(dz - dq*p_q[:,1])

reg_y = linear_model.LinearRegression()
reg_y.fit(p_q[:,0].reshape(-1,1), dy)

reg_z = linear_model.LinearRegression()
reg_z.fit(p_q[:,1].reshape(-1,1), dz)

y0 = reg_y.intercept_
z0 = reg_z.intercept_
dp = reg_y.coef_[0]
dq = reg_z.coef_[0]

ly = y0 + dp*p_q[:,0]
lz = z0 + dq*p_q[:,1]
lylz = y0*z0 + dp*dq*np.prod(p_q, axis=1) + z0*dp*p_q[:,0] + y0*dq*p_q[:,1]
assert np.allclose(ly*lz, lylz)

## Construct model with just P, Q
feat_pq = np.zeros_like(feat)
feat_pq[:,0] = np.prod(p_q, axis=1)
feat_pq[:,1:] = p_q
reg_pq = linear_model.LinearRegression()
reg_pq.fit(feat, energies_bulk)

reg_pq.intercept_ = alpha1*y0*z0 + alpha2*y0 + alpha3*z0
reg_pq.coef_[0] = alpha1*dp*dq
reg_pq.coef_[1] = dp*(alpha1*z0 + alpha2)
reg_pq.coef_[2] = dq*(alpha1*y0 + alpha3)



