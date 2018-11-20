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
bb = np.arange(-180,181,4)
for fname in fnames:
    this_mu = float(fname.split('_')[-1]) / 10.0
    payload_arr = np.load('{}/data_arr.npz'.format(fname))['arr_0']
    phi_vals = payload_arr[:,0]
    psi_vals = payload_arr[:,1]
    ntwid_dat = payload_arr[:,2]
    nreg_dat = payload_arr[:,3]
    weights = payload_arr[:,4]

    hist, bb, bb = np.histogram2d(phi_vals, psi_vals, bb, weights=weights)
    loghist = -np.log(hist)
    loghist -= loghist.min()

    hist_phi, bb = np.histogram(phi_vals, bb, weights=weights)
    loghist_phi = -np.log(hist_phi)
    loghist_phi -= loghist_phi.min()

    plt.plot(bb[:-1], loghist_phi)
    plt.savefig('g_phi_{:03g}'.format(this_mu*10))

    avg_dat = np.loadtxt('{}/averages.dat'.format(fname))
    averages.append(avg_dat)

    min_phi = -4
    max_phi = 10
    phi_barrier_mask = (phi_vals > min_phi) & (phi_vals < max_phi) 
    phi_barrier_logweights = -np.log(weights[phi_barrier_mask])
    #phi_barrier_logweights -= phi_barrier_logweights.min()

    barrier = -np.log(np.exp(-phi_barrier_logweights).sum())
    barrier_with_alpha = np.append(barrier_with_alpha, barrier)
    mu_vals = np.append(mu_vals, beta*this_mu)

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

fig, ax = plt.subplots(figsize=(5.5,5))
ax.plot(mu_vals, barrier_with_alpha, '-ok', markersize=12, linewidth=6)
ax.set_xlabel(r'$\beta \alpha$')
ax.set_ylabel(r'$\beta F_{\rm{max}}$')
ax.set_xticks([0,2,4,6,8])
fig.tight_layout()
fig.savefig('barrier.pdf', transparent=True)
plt.close('all')

averages = np.array(averages)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(mu_vals, averages[:,0], '-ok', markersize=12, linewidth=6)
shown_indices = [0,7,-1]
ax.plot(mu_vals[shown_indices], averages[shown_indices, 0], 'or', markersize=12)
ax.set_xlabel(r'$\beta \alpha$')
ax.set_ylabel(r'$\langle N_v \rangle_\alpha$')
ax.set_xticks([0,2,4,6,8])
fig.tight_layout()
fig.savefig('N_v_phi.pdf', transparent=True)



