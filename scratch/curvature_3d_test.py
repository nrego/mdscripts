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

buff = 2
res = 0.05
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

XX, YY, ZZ = np.meshgrid(grid_pts, grid_pts, grid_pts, indexing='ij')

# atoms per A^3
vol = (4/3.)*np.pi*(1.6)**3
density = 1 / vol

rho = np.zeros_like(XX)
phix = phi(XX, sig, sig**2, rcut, rcut**2)
phiy = phi(YY, sig, sig**2, rcut, rcut**2)
phiz = phi(ZZ, sig, sig**2, rcut, rcut**2)

rho += phix*phiy*phiz
rho /= density

# surface threshold
s = 0.5
verts, faces, normals, values = measure.marching_cubes_lewiner(rho, s, spacing=(1,1,1))
verts = verts*res + min_pt

ax = plt.gca(projection='3d')
ax.scatter(verts[:,0], verts[:,1], verts[:,2])
plt.show()

## Calculate local curvature ##

rho_x, rho_y, rho_z = np.gradient(rho)

# Second partial derivs (terms in the hessian)
rho_xx, rho_xy, rho_xz = np.gradient(rho_x)
rho_yx, rho_yy, rho_yz = np.gradient(rho_z)
rho_zx, rho_zy, rho_zz = np.gradient(rho_z)

trH = rho_xx+rho_yy+rho_zz
del_rho_sq = rho_x**2 + rho_y**2 + rho_z**2

# Mean curvature for all isosurfaces
#k_mean_rho = rho_x*rho_x*rho_xx + rho_y*rho_y*rho_yy + rho_z*rho_z*rho_zz + 2*rho_x*rho_y*rho_xy + 2*rho_x*rho_z*rho_xz + 2*rho_y*rho_z*rho_yz
k_mean_rho = rho_x*rho_x*rho_xx + rho_y*rho_y*rho_yy + rho_z*rho_z*rho_zz + rho_x*rho_y*(rho_xy+rho_yx) + rho_x*rho_z*(rho_xz+rho_zx) + rho_y*rho_z*(rho_yz+rho_zy)
k_mean_rho -= trH*del_rho_sq

k_mean_rho /= 2*del_rho_sq**1.5

k_mean_rho /= res

## Plot slice where z = 0
ax = plt.gca()
midpt_idx = grid_pts.size // 2
pcm = ax.pcolormesh(XX[:,:,midpt_idx], YY[:,:,midpt_idx], rho[:,:,midpt_idx], cmap='hot', norm=plt.Normalize(0,1))
plt.colorbar(pcm)
plt.show()

ax = plt.gca()
pcm = ax.pcolormesh(XX[:,:,midpt_idx], YY[:,:,midpt_idx], k_mean_rho[:,:,midpt_idx], cmap='hot', norm=plt.Normalize(0,1))
plt.colorbar(pcm)
plt.show()

## Find voxels at isosurface ##

surf = (rho > s).astype(float)
surf_x, surf_y, surf_z = np.gradient(surf)

isosurf_mask = (surf_x != 0) | (surf_y != 0) | (surf_z != 0)


isosurf_curvature_mean = k_mean_rho.copy()
isosurf_curvature_mean[~isosurf_mask] = 0
isosurf_curv_vals = isosurf_curvature_mean[isosurf_mask]

pts_all = np.vstack([XX[isosurf_mask].ravel(), YY[isosurf_mask].ravel(), ZZ[isosurf_mask].ravel()]).T

norm = plt.Normalize(-1,1)

ax = plt.gca(projection='3d')
ax.scatter(pts_all[:,0], pts_all[:,1], pts_all[:,2], c=isosurf_curv_vals, norm=norm, cmap='seismic')
plt.show()