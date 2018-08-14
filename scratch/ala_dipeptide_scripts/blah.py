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

from mdtools import dr

beta = 1/(k*300)

mpl.rcParams.update({'axes.labelsize': 60})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 40})

def min_dist(center,x):
    dx = np.abs(x-center)
     
    return np.amin([dx, 360-dx], axis=0)

ds = dr.loadPhi('equil/equil_run/phiout.dat')
binbounds = np.arange(-180,187,4)

bc = (binbounds[:-1] + binbounds[1:]) / 2.0

payload_arr = np.load('data_arr.npz')['arr_0']

phi_vals = payload_arr[:,0]
psi_vals = payload_arr[:,1]
ntwid_dat = payload_arr[:,2]
nreg_dat = payload_arr[:,3]
weights = payload_arr[:,4]

hist = histnd(np.array([phi_vals, psi_vals]).T, [binbounds, binbounds], weights=weights)

loghist = -np.log(hist)
loghist -= loghist.min()


extent = (-180,180,-180,180)
vmin, vmax = 0, 16
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


fig = plt.figure(figsize=(9,7))
#fig = plt.figure()
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

ax.set_title(r'$\beta \phi={:0.2f}$'.format(beta*ds.phi))
fig.tight_layout()
#plt.savefig('phi_new_{:03g}'.format(ds.phi*10))