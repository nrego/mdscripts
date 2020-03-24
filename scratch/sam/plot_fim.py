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
from sklearn import datasets, linear_model


from IPython import embed


def get_mse(reg, Z, y):
    y_hat = reg.predict(Z)

    return np.mean((y - y_hat)**2)

ds = np.load('sam_pattern_06_06.npz')
ds_bulk = np.load('sam_pattern_bulk_pure.npz')



print('\nExtracting sam data...')
p = q = 6
bulk_idx = np.where((ds_bulk['pq'] == (p,q)).all(axis=1))[0].item()
bulk_e = ds_bulk['energies'][bulk_idx]

print('  bulk energy: {:.2f}'.format(bulk_e))

states = ds['states']
energies = ds['energies']
errs = ds['err_energies']

dg_bind = energies - bulk_e

feat_vec = np.zeros((states.size, 4))

for i, state in enumerate(states):
    feat_vec[i,...] = 1, state.k_o, state.n_oo, state.n_oe

reg = linear_model.LinearRegression(fit_intercept=False)

int_vals = np.arange(-145, -109.99, 0.1)
a1_vals = np.arange(2, 6.01, 0.01)

## M1 ##
Z = feat_vec[:,:2]
reg.fit(Z, dg_bind)
p0 = reg.coef_.copy()
u, v = np.linalg.eig(np.dot(Z.T, Z))
v1 = v[:,0] + p0
v2 = v[:,1] + p0

ff, aa1 = np.meshgrid(int_vals, a1_vals, indexing='ij')
vals = np.zeros_like(ff)
vals[:] = np.nan

for i in range(vals.shape[0]):
    reg.coef_[0] = int_vals[i]
    for j in range(vals.shape[1]):
        reg.coef_[1] = a1_vals[j]

        vals[i,j] = get_mse(reg, Z, dg_bind)


pc = plt.pcolormesh(ff, aa1, vals, norm=plt.Normalize(40,100), cmap='jet')
plt.colorbar(pc)
plt.plot([p0[0], v1[0]], [p0[1], v1[1]], 'k-')
plt.plot([p0[0], v2[0]], [p0[1], v2[1]], 'k-')
plt.xlabel(r'$f_0$')
plt.ylabel(r'$\alpha_1$')
plt.show()

