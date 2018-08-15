from __future__ import division, print_function

import numpy as np
import matplotlib
mpl = matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
from scipy.optimize import minimize


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

ds = dr.loadPhi('equil/equil_run/phiout.dat')
this_mu = ds.phi

mpl.rcParams.update({'axes.labelsize': 60})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 40})

# For Phi/Psi angles
binbounds = np.arange(-180,187,4)

payload_arr = np.load('data_arr.npz')['arr_0']

phi_vals = payload_arr[:,0]
psi_vals = payload_arr[:,1]
ntwid_dat = payload_arr[:,2]
nreg_dat = payload_arr[:,3]
weights = payload_arr[:,4]

avg_n = np.dot(nreg_dat, weights)
avg_n_sq = np.dot(nreg_dat**2, weights)
var_n = avg_n_sq - avg_n**2
avg_ntwid = np.dot(ntwid_dat, weights)
avg_ntwid_sq = np.dot(ntwid_dat**2, weights)
var_ntwid = avg_ntwid_sq - avg_ntwid**2


#hist = histnd(np.array([phi_vals, psi_vals]).T, [binbounds, binbounds], weights=weights)
hist, bb, bb = np.histogram2d(phi_vals, psi_vals, binbounds, weights=weights)
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
plt.savefig('mu_conf_{:03g}'.format(this_mu*10))

avg_arr = np.array([avg_n, avg_ntwid, var_n, var_ntwid])
np.savetxt('averages.dat', avg_arr)


