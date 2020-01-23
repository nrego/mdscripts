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


def extract_probe_vol(umbr_path):
    with open(umbr_path, 'r') as fin:
        lines = fin.readlines()

    min_x, min_y, min_z, max_x, max_y, max_z = [float(s) for s in lines[3].split()]

    return (np.round(max_x-min_x, 4), np.round(max_y-min_y, 4), np.round(max_z-min_z, 4))

def extract_p_q(fname):
    splits = [int(s) for s in fname.split('/')[0].split('P')[1].split('_')]

    try:
        p,q = splits
    except ValueError:
        p = q = splits[0]

    return p,q

fnames = np.array(sorted(glob.glob('P*/PvN.dat')))

idx_to_remove = np.where((fnames=='P6/PvN.dat') | (fnames=='P4_9/PvN.dat') | (fnames=='P4/PvN.dat'))[0]
addl_fnames = fnames[idx_to_remove][::-1]
fnames = np.concatenate((np.delete(fnames, idx_to_remove), addl_fnames))

n_dat = len(fnames)

energies = np.zeros(n_dat)
errs = np.zeros_like(energies)
vols = np.zeros_like(energies)
sas = np.zeros_like(energies)

myfeat = np.zeros((n_dat, 3))

p_q = np.zeros_like(myfeat)

dq = 0.5
dp = 0.5*np.sqrt(3)*dq

for i, fname in enumerate(fnames):
    dirname = os.path.dirname(fname)

    dx, dy, dz = extract_probe_vol('{}/umbr.conf'.format(dirname))
    dat = np.loadtxt(fname)
    this_ener = dat[0,1]
    this_err  = dat[0,2]

    energies[i] = this_ener
    errs[i] = this_err
    vols[i] = dx*dy*dz
    sas[i] = dy*dz + dx*(dy+dz)

    myfeat[i] = dy*dz, dy, dz

    p, q = extract_p_q(fname)

    p_q[i] = p*q, p, q


y0 = np.mean(myfeat[:,1] - dp * p_q[:,1])
z0 = np.mean(myfeat[:,2] - dq * p_q[:,2])

e_min = energies[0]
perf_mse_subvol, err_subvol, xvals, fit, reg_subvol = fit_leave_one(myfeat[:,0].reshape(-1,1)-myfeat.min(), energies-e_min, fit_intercept=False)
alpha1 = reg_subvol.coef_[0]
perf_mse_pq, err_pq, xvals, fit, reg_pq = fit_leave_one(p_q, energies-e_min, fit_intercept=False)

lylz = y0*z0 + dp*z0*p_q[:,1] + dq*y0*p_q[:,2] + dp*dq*p_q[:,0]


reg = linear_model.LinearRegression()
reg.fit(p_q, energies)
reg.intercept_ = e_min
reg.coef_ = alpha1 * np.array([(dp*dq, dp*z0, dq*y0)])


