from __future__ import division, print_function

import westpa
from fasthist import histnd, normhistnd
import numpy as np
import matplotlib
mpl = matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
from whamutils import gen_U_nm, kappa, grad_kappa, gen_pdist
#import visvis as vv

from IPython import embed

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from mdtools import dr

import os
import glob


mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 20})

ntwid_forw = np.loadtxt('../phi_sims/forw/out.dat')
ntwid_revr = np.loadtxt('../phi_sims/revr/out.dat')
ntwid_revr2 = np.loadtxt('../phi_sims/revr2/out.dat')

err_ntwid_forw = np.loadtxt('../phi_sims/forw/ntwid_err.dat')
err_ntwid_revr = np.loadtxt('../phi_sims/revr/ntwid_err.dat')
err_ntwid_revr2 = np.loadtxt('../phi_sims/revr2/ntwid_err.dat')

fnames = sorted(glob.glob("mu_*/data_arr.npz"))


beta = 1/(0.0083144598*300)
phi_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0])
phi_vals *= beta

# Number of INDUS windows (each Phi/Psi landscape already unbiased)
n_windows = len(fnames)
assert phi_vals.size == n_windows

avg_ntwid = []
avg_ntwid_left = []
avg_ntwid_right = []

binbounds_phi = np.arange(-180,187,4)
binbounds_psi = binbounds_phi.copy()
binbounds_n = np.arange(0,81,1)

bc_phi = (binbounds_phi[:-1]+binbounds_phi[1:])/2.0
bc_psi = (binbounds_psi[:-1]+binbounds_psi[1:])/2.0
bc_n = (binbounds_n[:-1]+binbounds_n[1:])/2.0

binbounds = [binbounds_phi, binbounds_psi, binbounds_n]

n_samples = np.zeros(n_windows, dtype=np.int)

for i, fname in enumerate(fnames):
    this_dataset = np.load(fname)['arr_0']
    n_obs = this_dataset.shape[0]
    n_samples[i] = n_obs
    del this_dataset

hist = np.zeros((binbounds_phi.size-1, binbounds_psi.size-1, binbounds_n.size-1), dtype=np.float64)

vmin, vmax = 0, 16
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
extent = (-180,180,-180,180)

forw_start = np.loadtxt('../phi_sims/forw/rama.xvg', comments=["@","#"], usecols=(0,1))
revr_start = np.loadtxt('../phi_sims/revr/rama.xvg', comments=["@","#"], usecols=(0,1))
revr2_start = np.loadtxt('../phi_sims/revr2/rama.xvg', comments=["@","#"], usecols=(0,1))

for i,fname in enumerate(fnames):

    this_phi = phi_vals[i] / beta

    this_forw = np.loadtxt('../phi_sims/forw/phi_{:03g}/rama.xvg'.format(this_phi*10), comments=["@","#"], usecols=(0,1))
    this_revr = np.loadtxt('../phi_sims/revr/phi_{:03g}/rama.xvg'.format(this_phi*10), comments=["@","#"], usecols=(0,1))
    this_revr2 = np.loadtxt('../phi_sims/revr2/phi_{:03g}/rama.xvg'.format(this_phi*10), comments=["@","#"], usecols=(0,1))

    # Phi_ang    Psi_ang   N_twid    N_reg   Weight
    this_dataset = np.load(fname)['arr_0']

    phis = this_dataset[:,0]
    psis = this_dataset[:,1]
    ntwids = this_dataset[:,2]
    nregs = this_dataset[:,3]
    weights = this_dataset[:,4]

    ## mask of vals with phi < 0 (left hand side of rama plot)
    left_phi_mask = phis < 0
    right_phi_mask = ~left_phi_mask

    left_weights = weights[left_phi_mask]
    left_weights /= left_weights.sum()
    right_weights = weights[right_phi_mask]
    right_weights /= right_weights.sum()

    avg_ntwid.append(np.dot(ntwids, weights))
    avg_ntwid_left.append(np.dot(ntwids[left_phi_mask], left_weights))
    avg_ntwid_right.append(np.dot(ntwids[right_phi_mask], right_weights))

    this_hist = histnd(np.array([phis,psis]).T, [binbounds[0], binbounds[1]], weights=weights)
    loghist = -np.log(this_hist)
    loghist -= loghist.min()
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
                   cmap=cm.nipy_spectral, norm=norm, aspect='auto')
    cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                      colors='k', linewidths=1.0)
    cb = plt.colorbar(im)
    ax.set_xlabel('$\Phi$')
    ax.set_ylabel('$\Psi$')
    plt.plot(this_forw[:,0], this_forw[:,1], 'o', label='Forward', color='r', alpha=0.7)
    plt.plot(this_revr[:,0], this_revr[:,1], 'x', label='Reverse', markeredgecolor='k', alpha=0.7)
    plt.plot(this_revr2[:,0], this_revr2[:,1], '+', label='Reverse2', markeredgecolor='g', alpha=0.7)

    plt.plot(forw_start[0], forw_start[1], 'o', label='Forward start', color='r', markeredgecolor='k', markersize=18)
    plt.plot(revr_start[0], revr_start[1], 'X', label='Reverse start', color='r', markeredgecolor='k', markersize=18)
    plt.plot(revr2_start[0], revr2_start[1], 'P', label='Reverse2 start', color='r', markeredgecolor='k', markersize=18)

    ax.set_title('$phi={}$'.format(this_phi), fontsize=30)
    ax.set_xlim(-180,100)
    plt.tight_layout()
    plt.legend()
    plt.savefig('phi_{}_rama.png'.format(this_phi))

    plt.clf()
    plt.plot(this_forw[:,0], '-o', label='Forward', color='r')
    plt.plot(this_revr[:,0], '-x', label='Reverse', color='k')
    plt.plot(this_revr2[:,0], '-+', label='Reverse2', color='g')
    plt.title('phi={}'.format(this_phi), fontsize=30)
    plt.xlabel('time (ps)')
    plt.ylabel('$\Phi$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('phi_{}_ts.png'.format(this_phi))

    plt.close('all')

    vals = np.array([phis, psis, nregs]).T

    n_obs = n_samples[i]

    # Shape: (n_obs, n_windows)
    this_bias_mat = np.dot(ntwids[:,np.newaxis], phi_vals[np.newaxis,:])

    q = logweights - this_bias_mat
    denom = np.exp(q)

    denom = np.dot(denom, n_samples)

    histnd(vals, binbounds, weights=weights/denom, out=hist, binbound_check=False)

avg_ntwid = np.array(avg_ntwid)
avg_ntwid_left = np.array(avg_ntwid_left)
avg_ntwid_right = np.array(avg_ntwid_right)

plt.plot(phi_vals/beta, avg_ntwid, '-o', label='umbrella')
plt.plot(phi_vals/beta, avg_ntwid_left, '--o', label='umbrella, $\Phi<0$')
plt.plot(phi_vals/beta, avg_ntwid_right, '--o', label='umbrella, $\Phi>0$')

plt.errorbar(phi_vals/beta, ntwid_forw[:,1], yerr=err_ntwid_forw, fmt='-o', label='Forward')
plt.errorbar(phi_vals/beta, ntwid_revr[:,1], yerr=err_ntwid_revr, fmt='-o', label='Reverse')
plt.errorbar(phi_vals/beta, ntwid_revr2[:,1], yerr=err_ntwid_revr2, fmt='-o', label='Reverse2')

plt.legend()
plt.show()

hist /= hist.sum()
# Plot each slice

loghist_all = -np.log(hist)
loghist_all -= loghist_all.min()

for i in range(63):
    plt.figure()
    ax = plt.gca()
    thishist = hist[:,:,i].copy()
    thishist /= thishist.sum()
    loghist = -np.log(thishist)
    loghist -= loghist.min()
    
    loghist[loghist==float('inf')] = vmax

    im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
                   cmap=cm.nipy_spectral, norm=norm, aspect='auto')
    cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                      colors='k', linewidths=1.0)
    cb = plt.colorbar(im)
    ax.set_title('$N_V={:02g}$'.format(i), fontsize=30)
    ax.set_xlim(-180,100)
    plt.tight_layout()
    plt.savefig('plot_{:02g}.png'.format(i))
    plt.close('all')

# Integrate out Psi angles to get Phi angles vs N
phi_hist = hist.sum(axis=1)
phi_hist /= phi_hist.sum(axis=0) # Normalize conditional on each value N
psi_hist = hist.sum(axis=0)
psi_hist /= psi_hist.sum(axis=0)

# Just from Phi = -180 to 0
left_psi_hist = hist[:45].sum(axis=0)
left_psi_hist /= left_psi_hist.sum(axis=0)

# Just from Phi = 0 to 180
right_psi_hist = hist[45:].sum(axis=0)
right_psi_hist /= right_psi_hist.sum(axis=0)

extent = (-180,180,binbounds_n[0],binbounds_n[-1])

vmin, vmax = 0,16
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


### N v PHI ###
extent = (-180,180,0,80)

plt.figure()
ax = plt.gca()
loghist = -np.log(phi_hist)
loghist -= np.nanmin(loghist)

im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)
ax.set_xlim(-180,100)
ax.set_ylim(0,70)
ax.set_xlabel(r'$\Phi$')
ax.set_ylabel(r'$N_V$')
ax.set_title(r'$\Phi$ (all)')
plt.tight_layout()
plt.savefig('N_v_phi.png')



### N v PSI (all) ###

extent = (0,80,-180,180)
plt.figure()
ax = plt.gca()
loghist = -np.log(psi_hist)
loghist -= np.nanmin(loghist)

im = ax.imshow(loghist, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)
ax.set_xlabel(r'$N_V$')
ax.set_xlim(0,70)
ax.set_ylabel(r'$\Psi$')
ax.set_title(r'$\Psi$ (all)')
plt.tight_layout()
plt.savefig('N_v_psi.png')

## Flip ##
extent = (-180,180,0,80)
plt.figure()
ax = plt.gca()
loghist = -np.log(psi_hist)
loghist -= np.nanmin(loghist)

im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)
ax.set_xlabel(r'$\Psi$')
ax.set_ylabel(r'$N_V$')
ax.set_ylim(0,70)
ax.set_title(r'$\Psi$ (all)')
plt.tight_layout()
plt.savefig('N_v_psi_flip.png')


## N v PSI (left) ##
extent = (0,80,-180,180)
fig = plt.figure()
ax = plt.gca()
loghist = -np.log(left_psi_hist)
loghist -= np.nanmin(loghist)

im = ax.imshow(loghist, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)
ax.set_xlabel(r'$N_V$')
ax.set_xlim(0,70)
ax.set_ylabel(r'$\Psi$')
ax.set_title(r'$\Psi$ ($\Phi=-180$ to $0$)')
plt.tight_layout()
plt.savefig('N_v_psi_left.png')

## Flipped ##
extent = (-180,180,0,80)
fig = plt.figure()
ax = plt.gca()
loghist = -np.log(left_psi_hist)
loghist -= np.nanmin(loghist)

im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)
ax.set_ylabel(r'$N_V$')
ax.set_ylim(0,70)
ax.set_xlabel(r'$\Psi$')
ax.set_title(r'$\Psi$ ($\Phi=-180$ to $0$)')
plt.tight_layout()
plt.savefig('N_v_psi_left_flip.png')


## N v PSI (right) ##
extent = (0,80,-180,180)
fig = plt.figure()
ax = plt.gca()
loghist = -np.log(right_psi_hist)
loghist -= np.nanmin(loghist)

im = ax.imshow(loghist, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)
ax.set_xlabel(r'$N_V$')
ax.set_xlim(0,70)
ax.set_ylabel(r'$\Psi$')
ax.set_title(r'$\Psi$ ($\Phi=0$ to $180$)')
plt.tight_layout()
plt.savefig('N_v_psi_right.png')

# Flip
extent = (-180,180,0,80)
fig = plt.figure()
ax = plt.gca()
loghist = -np.log(right_psi_hist)
loghist -= np.nanmin(loghist)

im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)
ax.set_ylabel(r'$N_V$')
ax.set_ylim(0,70)
ax.set_xlabel(r'$\Psi$')
ax.set_title(r'$\Psi$ ($\Phi=0$ to $180$)')
plt.tight_layout()
plt.savefig('N_v_psi_right_flip.png')