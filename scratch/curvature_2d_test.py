from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
import logging

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import MDAnalysis

import os, glob
import cPickle as pickle
from scipy.integrate import cumtrapz

import math

from rhoutils import phi_1d

from skimage import measure

from scipy.interpolate import CubicSpline

## Testing ##
gaus = lambda x_sq, sig_sq: np.exp(-x_sq/(2*sig_sq))

def plot_3d_scatter(x, y, z, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca(projection='3d')

    ax.scatter(x,y,z,**kwargs)

    return ax

def plot_3d_surf(XX, YY, ZZ, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca(projection='3d')

    ax.plot_surface(XX,YY,ZZ,**kwargs)

    return ax

def phi(r, sig, sig_sq, rcut, rcut_sq):

    r = np.array(r, ndmin=1)
    r_sq = r**2

    out_bound = r_sq > rcut_sq

    pref = 1 / ( np.sqrt(np.pi*2*sig_sq) * math.erf(rcut/np.sqrt(2*sig_sq)) - 2*rcut*np.exp(-rcut_sq/2*sig_sq) )

    ret_arr = pref * (gaus(r_sq, sig_sq) - gaus(rcut_sq, sig_sq))
    ret_arr[out_bound] = 0

    return ret_arr


mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 30})

homedir = os.environ['HOME']

## Units in A
pt = np.array([0.,0.,0.])

buff = 5
res = 0.01
sig = 1
rcut = 5

min_pt = -buff
max_pt = buff

grid_pts = np.arange(min_pt, max_pt+res, res, dtype=np.float32)

phi_vals = phi(grid_pts, sig, sig**2, rcut, rcut**2)

#ax = plt.gca()
#ax.plot(grid_pts, phi_vals)
#ax.set_xlim(min_pt, max_pt)
#plt.show()

XX, YY = np.meshgrid(grid_pts, grid_pts, indexing='ij')

# atoms per A^2
vol = np.pi*(1.6)**2
density = 1 / vol

rho = np.zeros_like(XX)
phix = phi(XX, sig, sig**2, rcut, rcut**2)
phiy = phi(YY, sig, sig**2, rcut, rcut**2)
rho += phix*phiy
rho /= density

# surface threshold
s = 0.5

cmap = cm.hot_r
norm = mpl.colors.Normalize(0, rho.max())

fig, ax = plt.subplots(figsize=(7,6))

pmesh = ax.pcolormesh(XX, YY, rho, cmap=cmap, norm=norm)
plt.colorbar(pmesh)

contour = measure.find_contours(rho, s)[0]
contour *= res
contour += min_pt

ax.plot(contour[:,0], contour[:,1], 'k--')

plt.show()


## Find curvature, etc of implicit surface from rho grid ##
rho_x, rho_y = np.gradient(rho)
rho_xx, rho_xy = np.gradient(rho_x)
rho_yx, rho_yy = np.gradient(rho_y)

# Curvature at any isocurve
k_rho = -( (rho_x**2)*rho_yy + (rho_y**2)*rho_xx - 2*rho_x*rho_y*rho_xy ) / (rho_x**2 + rho_y**2)**1.5
k_rho = np.ma.masked_invalid(k_rho)
k_rho /= res

## Make binary surface - rho is either >= 0.5 (1) or < 0.5 (0)
surf = (rho >= s).astype(float)
surf_x, surf_y = np.gradient(surf)
surf_xx, surf_xy = np.gradient(surf_x)
surf_yx, surf_yy = np.gradient(surf_y)

# curvature at each point #
k_surf = ( (surf_x**2)*surf_yy + (surf_y**2)*surf_xx - 2*surf_x*surf_y*surf_xy ) / (surf_x**2 + surf_y**2)**1.5
k_surf = np.ma.masked_invalid(k_surf)

rng_pt = 1
norm = plt.Normalize(0, rng_pt)

ax = plt.gca()
pmesh = ax.pcolormesh(XX, YY, k_rho, cmap='hot', norm=norm)
plt.colorbar(pmesh)
plt.show()

### plot curvature for isocurve ###

# Mask of isocurve values
isocurve_mask = (surf_x != 0) | (surf_y != 0)
# Curvature at each point of isocurve for s
isocurve_curvature = k_rho.copy()
isocurve_curvature[~isocurve_mask] = 0

ax = plt.gca()
ax.pcolormesh(XX, YY, isocurve_curvature, cmap='seismic', norm=plt.Normalize(-1,1))
