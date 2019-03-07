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
pt = np.array([0,0,0])

buff = 4
res = 0.1

min_pt = -10
max_pt = 10

grid_pts = np.arange(min_pt, max_pt+res, res)


# volume taken up by single point with radius=1.6 A
vol = np.pi*(1.6)**3
# its density
density = 1/vol
sig = 1
rcut = 6

s = 0.5

phivals = phi(grid_pts, sig, sig**2, rcut, rcut**2)
plt.plot(grid_pts, phivals)
plt.show()


XX, YY, ZZ = np.meshgrid(grid_pts, grid_pts, grid_pts, indexing='ij')

rho = np.zeros_like(XX)

phix = phi(XX, sig, sig**2, rcut, rcut**2)
phiy = phi(YY, sig, sig**2, rcut, rcut**2)
phiz = phi(ZZ, sig, sig**2, rcut, rcut**2)

rho += phix*phiy*phiz
rho /= density

verts, faces, normals, values = measure.marching_cubes_lewiner(rho, s, spacing=(1,1,1))

mesh = Poly3DCollection(verts[faces]+min_pt)
mesh.set_edgecolor('k')
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.add_collection(mesh)
ax.scatter(verts[:,0]*res + min_pt, verts[:,1]*res + min_pt, verts[:,2]*res + min_pt)
#ax.set_xlim(min_pt, max_pt)
#ax.set_ylim(min_pt, max_pt)
#ax.set_zlim(min_pt, max_pt)
#ax.scatter(pos[:,0], pos[:,1], pos[:,2], 'k')

## Find curvature, etc of implicit surface from rho grid ##
rho_x, rho_y, rho_z = np.gradient(rho)
rho_xx, rho_xy, rho_xz = np.gradient(rho_x)
rho_yx, rho_yy, rho_yz = np.gradient(rho_y)
rho_zx, rho_zy, rho_zz = np.gradient(rho_z)

# terms for adjoint hessian # 

# as in:
###   h_xx  h_xy  h_xz   ###
###   h_yx  h_yy  h_yz   ###
###   h_zx  h_zy  h_zz   ###

h_xx = rho_yy*rho_zz - rho_yz*rho_zy
h_xy = rho_yz*rho_zx - rho_yx*rho_zz
h_xz = rho_yx*rho_zy - rho_yy*rho_zx

h_yx = rho_xz*rho_zy - rho_xy*rho_zz
h_yy = rho_xx*rho_zz - rho_xz*rho_zx
h_yz = rho_xy*rho_zx - rho_xx*rho_zy

h_zx = rho_xy*rho_yz - rho_xz*rho_yy
h_zy = rho_yx*rho_xz - rho_xx*rho_yz
h_zz = rho_xx*rho_yy - rho_xy*rho_yx


k_gaus_rho = ((rho_x**2)*h_xx + rho_x*rho_y*h_yx + rho_x*rho_z*h_xz + rho_x*rho_y*h_xy + (rho_y**2)*h_yy + rho_y*rho_z*h_yz + rho_x*rho_z*h_xz + rho_y*rho_z*h_yz + (rho_z**2)*h_zz)
k_gaus_rho /= (rho_x**2 + rho_y**2 + rho_z**2)**2


surf = (rho > s).astype(float)
surf_x, surf_y, surf_z = np.gradient(surf)
surf_xx, surf_xy, surf_xz = np.gradient(surf_x)
surf_yx, surf_yy, surf_yz = np.gradient(surf_y)
surf_zx, surf_zy, surf_zz = np.gradient(surf_z)

# terms for adjoint hessian # 

# as in:
###   h_xx  h_xy  h_xz   ###
###   h_yx  h_yy  h_yz   ###
###   h_zx  h_zy  h_zz   ###

h_xx = surf_yy*surf_zz - surf_yz*surf_zy
h_xy = surf_yz*surf_zx - surf_yx*surf_zz
h_xz = surf_yx*surf_zy - surf_yy*surf_zx

h_yx = surf_xz*surf_zy - surf_xy*surf_zz
h_yy = surf_xx*surf_zz - surf_xz*surf_zx
h_yz = surf_xy*surf_zx - surf_xx*surf_zy

h_zx = surf_xy*surf_yz - surf_xz*surf_yy
h_zy = surf_yx*surf_xz - surf_xx*surf_yz
h_zz = surf_xx*surf_yy - surf_xy*surf_yx

k_gaus_surf = ((surf_x**2)*h_xx + surf_x*surf_y*h_yx + surf_x*surf_z*h_xz + surf_x*surf_y*h_xy + (surf_y**2)*h_yy + surf_y*surf_z*h_yz + surf_x*surf_z*h_xz + surf_y*surf_z*h_yz + (surf_z**2)*h_zz)
k_gaus_surf /= (surf_x**2 + surf_y**2 + surf_z**2)**2

surf_mask = ~np.ma.masked_invalid(k_gaus_surf.ravel()).mask

#fig = plt.figure()
#ax = fig.gca(projection='3d')

all_pts = np.vstack((XX.ravel(), YY.ravel(), ZZ.ravel())).T
surf_pts = all_pts[surf_mask]

# Local (gaussian) curvature of each surface point
k_pts = k_gaus_surf.ravel()[surf_mask]

ax.scatter(surf_pts[:,0], surf_pts[:,1], surf_pts[:,2], color='k', alpha=0.5)
plt.show()

## Now plot surface points and color them by local curvature

rng_pt = np.abs(k_pts).max()
norm = plt.Normalize(-rng_pt, rng_pt)
colors = cm.seismic(norm(k_pts))

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(surf_pts[:,0], surf_pts[:,1], surf_pts[:,2], c=k_pts, norm=norm, cmap='seismic')








