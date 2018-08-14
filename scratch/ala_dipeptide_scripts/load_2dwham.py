from __future__ import division, print_function



import numpy as np
import matplotlib
mpl = matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
from scipy.optimize import minimize
from whamutils import kappa, grad_kappa, hess_kappa, gen_data_logweights
import pymbar
#import visvis as vv

#from IPython import embed

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from mdtools import dr

from constants import k

import os
import glob

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 20})

def min_dist(center,x):
    dx = np.abs(x-center)
     
    return np.amin([dx, 360-dx], axis=0)


fnames = sorted(glob.glob('*/*/rama.xvg'))

n_windows = len(fnames)

##TODO: Encapsulate this in a new data reader!
phi_vals = []
psi_vals = []

uncorr_phi_vals = []
uncorr_psi_vals = []

phi_kappas = []
phi_stars = []
psi_kappas = []
psi_stars = []

n_samples = []
uncorr_n_samples = []

nreg_dat = []
ntwid_dat = []
uncorr_nreg_dat = []
uncorr_ntwid_dat = []

start = 500
this_mu = None
for fname in fnames:
    rama = np.loadtxt(fname, usecols=(0,1), comments=['@','#'])
    dirname = os.path.dirname(fname)
    ds = dr.loadPhi('{}/phiout.dat'.format(dirname))
    if this_mu is None:
        this_mu = ds.phi
    else:
        assert this_mu == ds.phi
    n_dat = ds.data[start::10]
    this_nreg_dat = np.array(n_dat['N'])
    this_ntwid_dat = np.array(n_dat['$\~N$'])

    with open("{}/topol.top".format(dirname), "r") as f:
        lines = f.readlines()
        phi_line = lines[-34].strip().split()
        psi_line = lines[-33].strip().split()

        phi_kappas.append(float(phi_line[-1]))
        phi_stars.append(float(phi_line[-2]))

        psi_kappas.append(float(psi_line[-1]))
        psi_stars.append(float(psi_line[-2]))

        print("dir: {}, mu: {}, kap_phi: {}, phi_star: {}, kap_psi: {}, psi_star: {}".format(dirname, ds.phi, float(phi_line[-1]), float(phi_line[-2]), float(psi_line[-1]), float(psi_line[-2])))


    this_phi = rama[start:, 0]
    this_psi = rama[start:, 1]

    this_phi_diff = min_dist(phi_stars[-1], this_phi)
    this_psi_diff = min_dist(psi_stars[-1], this_psi)
    
    if fname.split('/')[0] == 'equil':
        g_inef = 1
    else:
        tau1 = pymbar.timeseries.integratedAutocorrelationTime(this_phi_diff)
        tau2 = pymbar.timeseries.integratedAutocorrelationTime(this_psi_diff)
        g_inef = int((max(tau1, tau2) * 2) + 1)

    n_sample = this_phi.size
    n_uncorr_sample = n_sample // g_inef
    rem = n_sample % g_inef

    this_uncorr_phi = this_phi[rem:].reshape(n_uncorr_sample, g_inef).mean(axis=1)
    this_uncorr_psi = this_psi[rem:].reshape(n_uncorr_sample, g_inef).mean(axis=1)
    this_uncorr_nreg_dat = this_nreg_dat[rem:].reshape(n_uncorr_sample, g_inef).mean(axis=1)
    this_uncorr_ntwid_dat = this_ntwid_dat[rem:].reshape(n_uncorr_sample, g_inef).mean(axis=1)

    uncorr_phi_vals.append(this_uncorr_phi)
    uncorr_psi_vals.append(this_uncorr_psi)
    uncorr_nreg_dat.append(this_uncorr_nreg_dat)
    uncorr_ntwid_dat.append(this_uncorr_ntwid_dat)

    print("  n_samples: {}, uncorr_n_samples: {}".format(n_sample, n_uncorr_sample))

    nreg_dat.append(n_dat['N'])
    ntwid_dat.append(n_dat['$\~N$'])
    phi_vals.append(this_phi)
    psi_vals.append(this_psi)
    n_samples.append(n_sample)
    uncorr_n_samples.append(n_uncorr_sample)
    dr.clearData()


phi_vals = np.concatenate(phi_vals)
psi_vals = np.concatenate(psi_vals)
uncorr_phi_vals = np.concatenate(uncorr_phi_vals)
uncorr_psi_vals = np.concatenate(uncorr_psi_vals)

nreg_dat = np.concatenate(nreg_dat)
ntwid_dat = np.concatenate(ntwid_dat)
uncorr_nreg_dat = np.concatenate(uncorr_nreg_dat)
uncorr_ntwid_dat = np.concatenate(uncorr_ntwid_dat)

assert phi_vals.size == psi_vals.size == ntwid_dat.size == nreg_dat.size

n_samples = np.array(n_samples)
uncorr_n_samples = np.array(uncorr_n_samples)

n_tot = phi_vals.size
uncorr_n_tot = uncorr_n_samples.sum()
assert n_samples.sum() == n_tot

beta = 1/(k * 300)

phi_kappas = np.array(phi_kappas)
phi_kappas = (phi_kappas * np.pi**2) / (180.)**2
phi_stars = np.array(phi_stars)
psi_kappas = np.array(psi_kappas)
psi_kappas = (psi_kappas * np.pi**2) / (180.)**2
psi_stars = np.array(psi_stars)

assert phi_kappas.size == psi_kappas.size == n_windows

bias_mat = np.zeros((n_tot, n_windows), dtype=np.float32)
uncorr_bias_mat = np.zeros((uncorr_n_tot, n_windows), dtype=np.float32)

for i in range(n_windows):
    phi_kappa = phi_kappas[i]
    phi_star = phi_stars[i]

    psi_kappa = psi_kappas[i]
    psi_star = psi_stars[i]

    bias_mat[:,i] = beta * ((phi_kappa*0.5)*(min_dist(phi_star,phi_vals))**2 + (psi_kappa*0.5)*(min_dist(psi_star,psi_vals))**2)
    uncorr_bias_mat[:,i] = beta * ((phi_kappa*0.5)*(min_dist(phi_star,uncorr_phi_vals))**2 + (psi_kappa*0.5)*(min_dist(psi_star,uncorr_psi_vals))**2)

binbounds = np.arange(-180,187,4)

bc = (binbounds[:-1] + binbounds[1:]) / 2.0

#hist = histnd(np.array([phi_vals, psi_vals]).T, [binbounds, binbounds])
hist, bb, bb = np.histogram2d(phi_vals, psi_vals, binbounds)

loghist = -np.log(hist)
loghist -= loghist.min()

# Save overlap plot

extent = (-180,180,-180,180)
vmin, vmax = 0, 8
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

fig = plt.figure()
ax = plt.gca()

im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)

ax.set_xlim(-180,100)
ax.set_ylim(-180,180)

ax.set_xlabel(r'$\Phi$')
ax.set_ylabel(r'$\Psi$')

ax.set_title(r'$\phi={}$ kJ/mol'.format(ds.phi))
fig.tight_layout()
plt.savefig('overlap_phi_{:03g}'.format(ds.phi*10))


### Perform WHAM 
print("doing WHAM with all data")
n_sample_diag = np.matrix( np.diag(n_samples / n_tot), dtype=np.float32)

ones_m = np.matrix(np.ones(n_windows,), dtype=np.float32).T
# (n_tot x 1) ones vector; n_tot = sum(n_k) total number of samples over all windows
ones_n = np.matrix(np.ones(n_tot,), dtype=np.float32).T

xweights = np.zeros(n_windows)

myargs = (bias_mat, n_sample_diag, ones_m, ones_n, n_tot)

ret = minimize(kappa, xweights[1:], args=myargs, method='L-BFGS-B', jac=grad_kappa)
f_ks = np.append(0, -ret['x'])

np.savetxt('f_ks.dat', f_ks)

### Get the unbiased histogram ###
logweights = gen_data_logweights(bias_mat, f_ks, n_samples)

weights = np.exp(logweights)
weights /= weights.sum()

hist, bb, bb = np.histogram2d(phi_vals, psi_vals, binbounds, weights=weights)

loghist = -np.log(hist)
loghist -= loghist.min()


extent = (-180,180,-180,180)
vmin, vmax = 0, 16
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)



fig = plt.figure()
ax = plt.gca()

im = ax.imshow(loghist.T, extent=extent, interpolation='nearest', origin='lower', alpha=0.75,
               cmap=cm.nipy_spectral, norm=norm, aspect='auto')
cont = ax.contour(loghist.T, extent=extent, origin='lower', levels=np.arange(vmin,vmax,1),
                  colors='k', linewidths=1.0)
cb = plt.colorbar(im)

ax.set_xlim(-180,100)
ax.set_ylim(-180,180)

ax.set_xlabel(r'$\Phi$')
ax.set_ylabel(r'$\Psi$')

ax.set_title(r'$\phi={}$ kJ/mol'.format(ds.phi))
fig.tight_layout()
plt.savefig('phi_all_{:03g}'.format(ds.phi*10))

# save it
payload_arr = np.dstack((phi_vals, psi_vals, ntwid_dat, nreg_dat, weights)).squeeze()

np.savez_compressed('data_arr', payload_arr)
