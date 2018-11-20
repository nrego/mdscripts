from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
from scipy.optimize import minimize

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import os
import glob

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 30})

# Run from directory e.g. mu_000
fnames = glob.glob('mu_*')

kT = (0.0083144598 * 300)
beta = 1/kT

averages = []
barrier_with_alpha = np.array([])
mu_vals = np.array([])
bb = np.arange(-180,185,4)
for fname in fnames:
    this_mu = float(fname.split('_')[-1]) / 10.0
    payload_arr = np.load('{}/data_arr.npz'.format(fname))['arr_0']
    phi_vals = payload_arr[:,0]
    psi_vals = payload_arr[:,1]
    ntwid_dat = payload_arr[:,2]
    nreg_dat = payload_arr[:,3]
    weights = payload_arr[:,4]
    weights /= weights.sum()

    hist, bb, bb = np.histogram2d(phi_vals, psi_vals, bb, weights=weights)
    loghist = -np.log(hist)
    loghist -= loghist.min()

    phi_assign = np.digitize(phi_vals, bins=bb) - 1
    psi_assign = np.digitize(psi_vals, bins=bb) - 1
    new_loghist = np.zeros_like(loghist)
    hyd_weight = np.zeros_like(loghist)

    for x in range(bb.size-1):
        phi_mask = phi_assign == x
        for y in range(bb.size-1):
            psi_mask = psi_assign == y

            mask = phi_mask & psi_mask

            if mask.sum() == 0:
                hyd_weight[x,y] = np.nan
                new_loghist[x,y] = np.nan

            else:
                new_loghist[x,y] = weights[mask].sum()
                this_wts = -np.log(weights[mask])
                this_wts -= this_wts.min()
                this_wts = np.exp(-this_wts)
                this_wts /= this_wts.sum()

                hyd_weight[x,y] = np.dot(nreg_dat[mask], this_wts)


    avg_dat = np.loadtxt('{}/averages.dat'.format(fname))
    averages.append(avg_dat)

    extent = (-180,180,-180,180)
    vmin, vmax = 0, 16
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)


    fig = plt.figure(figsize=(9,7))
    #fig = plt.figure()
    ax = plt.gca()

    im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
                   cmap=cm.nipy_spectral, norm=norm, aspect='auto')
    cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                      colors='k', linewidths=1.0)
    cb = plt.colorbar(im)
    ax.vlines([min_phi, max_phi], -180, 180)

    ax.set_xlim(-180,100)
    ax.set_ylim(-180,180)

    #ax.set_xlabel(r'$\Phi$')
    #ax.set_ylabel(r'$\Psi$')

    #ax.set_title(r'$\beta \alpha={:0.2f}$'.format(beta*this_mu))
    fig.tight_layout()
    plt.savefig('fig_mu_conf_{:03g}'.format(this_mu*10), transparent=True)

    plt.close('all')





