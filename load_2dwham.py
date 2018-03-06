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


def min_dist(center,x):
    dx = np.abs(x-center)
     
    return np.amin([dx, 360-dx], axis=0)


fnames = sorted(glob.glob('*/*/rama.xvg'))

n_windows = len(fnames)

##TODO: Encapsulate this in a new data reader!
phi_vals = []
psi_vals = []

phi_kappas = []
phi_stars = []
psi_kappas = []
psi_stars = []

n_samples = []

nreg_dat = []
ntwid_dat = []

start = 500

for fname in fnames:
    rama = np.loadtxt(fname, usecols=(0,1), comments=['@','#'])
    dirname = os.path.dirname(fname)
    ds = dr.loadPhi('{}/phiout.dat'.format(dirname))
    n_dat = ds.data[start::10]

    with open("{}/topol.top".format(dirname), "r") as f:
        lines = f.readlines()
        phi_line = lines[-34].strip().split()
        psi_line = lines[-33].strip().split()

        phi_kappas.append(float(phi_line[-1]))
        phi_stars.append(float(phi_line[-2]))

        psi_kappas.append(float(psi_line[-1]))
        psi_stars.append(float(psi_line[-2]))

        print("dir: {}, mu: {}, kap_phi: {}, phi_star: {}, kap_psi: {}, psi_star: {}".format(dirname, ds.phi, float(phi_line[-1]), float(phi_line[-2]), float(psi_line[-1]), float(psi_line[-2])))

    
    nreg_dat.append(n_dat['N'])
    ntwid_dat.append(n_dat['$\~N$'])
    phi_vals.append(rama[start:, 0])
    psi_vals.append(rama[start:, 1])
    n_samples.append(rama[start:].shape[0])
    dr.clearData()


phi_vals = np.concatenate(phi_vals)
psi_vals = np.concatenate(psi_vals)

nreg_dat = np.concatenate(nreg_dat)
ntwid_dat = np.concatenate(ntwid_dat)

assert phi_vals.size == psi_vals.size == ntwid_dat.size == nreg_dat.size

n_samples = np.array(n_samples)

n_tot = phi_vals.size
assert n_samples.sum() == n_tot

beta = 1/(8.3144598e-3 * 300)

phi_kappas = np.array(phi_kappas)
phi_kappas = (phi_kappas * np.pi**2) / (180.)**2
phi_stars = np.array(phi_stars)
psi_kappas = np.array(psi_kappas)
psi_kappas = (psi_kappas * np.pi**2) / (180.)**2
psi_stars = np.array(psi_stars)

assert phi_kappas.size == psi_kappas.size == n_windows

bias_mat = np.zeros((n_tot, n_windows), dtype=np.float32)

for i in range(n_windows):
    phi_kappa = phi_kappas[i]
    phi_star = phi_stars[i]

    psi_kappa = psi_kappas[i]
    psi_star = psi_stars[i]

    bias_mat[:,i] = beta * ((phi_kappa*0.5)*(min_dist(phi_star,phi_vals))**2 + (psi_kappa*0.5)*(min_dist(psi_star,psi_vals))**2)

binbounds = np.arange(-180,187,4)

bc = (binbounds[:-1] + binbounds[1:]) / 2.0

hist = histnd(np.array([phi_vals, psi_vals]).T, [binbounds, binbounds])

loghist = -np.log(hist)
loghist -= loghist.min()


n_sample_diag = np.matrix( np.diag(n_samples / n_tot), dtype=np.float32)

ones_m = np.matrix(np.ones(n_windows,), dtype=np.float32).T
# (n_tot x 1) ones vector; n_tot = sum(n_k) total number of samples over all windows
ones_n = np.matrix(np.ones(n_tot,), dtype=np.float32).T

xweights = np.zeros(n_windows)

myargs = (bias_mat, n_sample_diag, ones_m, ones_n, n_tot)

logweights = fmin_bfgs(kappa, xweights[1:], fprime=grad_kappa, args=myargs)[0]
logweights = -np.append(0, logweights)

q = logweights - bias_mat
denom = np.dot(np.exp(q), n_samples)
weights = 1/denom
weights /= weights.sum()

hist = histnd(np.array([phi_vals, psi_vals]).T, [binbounds, binbounds], weights=weights)

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


