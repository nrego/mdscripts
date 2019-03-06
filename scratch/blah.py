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

def construct_local_curve(contour, pt_idx, window=100):
    n_pts_tot = contour.shape[0]
    assert type(window) is int and type(pt_idx) is int, "Pt_idx and window must both be ints"
    assert pt_idx >= 0 and pt_idx < n_pts_tot, "Invalid pt_idx"
    assert window < n_pts_tot / 2, "window size is larger than n_pts_tot/2"
    
    ## point in middle of curve ##
    if pt_idx - window >= 0 and pt_idx + window < n_pts_tot:
        return contour[pt_idx - window:pt_idx+window+1].copy()

    if pt_idx - window < 0:
        remainder = np.abs(pt_idx - window)
        pts_under = contour[-remainder:].copy()
        pts_over = contour[:pt_idx+window+1].copy()

        return np.vstack((pts_under, pts_over))

    if pt_idx + window >= n_pts_tot:
        remainder = pt_idx + window + 1 - n_pts_tot
        pts_under = contour[pt_idx-window:n_pts_tot].copy()
        pts_over = contour[:remainder].copy()

        return np.vstack((pts_under, pts_over))
        

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
pos = np.loadtxt('pos.dat'.format(homedir), dtype=np.float32)
buried_mask = np.loadtxt('{}/simulations/ppi_analysis/2b97/bound/buried_mask.dat'.format(homedir), dtype=bool)
surf_mask = ~buried_mask

buff = 4
res = 0.1

min_pt = np.floor(pos.min()) - buff
max_pt = np.ceil(pos.max()) + buff

grid_pts = np.arange(min_pt, max_pt+res, res, dtype=np.float32)

dx, dy = np.meshgrid(grid_pts, grid_pts, indexing='ij')

# atoms per A^2
rho_prot = 0.040
rho = np.zeros_like(dx)
sig = 1
rcut = 10

# surface threshold
s = 0.5

circles = []
for i in range(pos.shape[0]):
    pt = pos[i]
    if surf_mask[i]:
        circles.append(Circle(pt, 1.6, fill=False))
    phix = phi(dx-pt[0], sig, sig**2, rcut, rcut**2)
    phiy = phi(dy-pt[1], sig, sig**2, rcut, rcut**2)

    rho += phix*phiy

rho /= rho_prot
cmap = cm.hot_r
norm = mpl.colors.Normalize(0, rho.max())

fig, ax = plt.subplots(figsize=(6,6))
collection = PatchCollection(circles)
collection.set_facecolors([0,0,0,0])
collection.set_edgecolors([0,0,0,1])

ax.pcolormesh(dx, dy, rho, cmap=cmap, norm=norm)
ax.add_collection(collection)

contour = measure.find_contours(rho, s)[0]
contour *= res
contour += min_pt

points = contour.reshape(-1,1,2)

segments = np.concatenate([points[:-1], points[1:]], axis=1)

cmap2 = cm.seismic
norm = plt.Normalize(0, segments.shape[0])
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(np.arange(segments.shape[0]))

ax.add_collection(lc)

plt.show()


## Find curvature, etc of implicit surface from rho grid ##
rho_x, rho_y = np.gradient(rho)
rho_xx, rho_xy = np.gradient(rho_x)
rho_yx, rho_yy = np.gradient(rho_y)

k_rho = -( (rho_x**2)*rho_yy + (rho_y**2)*rho_xx - 2*rho_x*rho_y*rho_xy ) / (rho_x**2 + rho_y**2)**1.5
#k_rho = np.ma.masked_invalid(k_rho)

## Make binary surface - rho is either >= 0.5 (1) or < 0.5 (0)
surf = (rho >= s).astype(float)
surf_x, surf_y = np.gradient(surf)
surf_xx, surf_xy = np.gradient(surf_x)
surf_yx, surf_yy = np.gradient(surf_y)

# curvature at each point #
k_surf = ( (surf_x**2)*surf_yy + (surf_y**2)*surf_xx - 2*surf_x*surf_y*surf_xy ) / (surf_x**2 + surf_y**2)**1.5
k_surf = np.ma.masked_invalid(k_surf)

# Mask of curve values
mask = k_surf.mask

#k_rho[mask] = np.nan
k_rho = np.ma.masked_invalid(k_rho)
rng_pt = 0.25 #np.abs(k_rho).max()
norm = plt.Normalize(-rng_pt, rng_pt)


fig, ax = plt.subplots(figsize=(7,6))

m = ax.pcolormesh(dx, dy, k_rho, norm=norm, cmap='seismic')
collection = PatchCollection(circles)
collection.set_facecolors([0,0,0,0])
collection.set_edgecolors([0,0,0,1])
ax.add_collection(collection)
cb = plt.colorbar(m)

ax.plot(contour[:,0], contour[:,1], 'k--')

plt.show()

## Find mean curvature for each atom - gaussian-distance weighted sum of its local curvature ##
mean_curvature = np.zeros(pos.shape[0])
rcut = 3
sig = 1
circles_colored = []
for i, pt in enumerate(pos):

    phix = phi(dx-pt[0], sig, sig**2, rcut, rcut**2)
    phiy = phi(dy-pt[1], sig, sig**2, rcut, rcut**2)
    wt = phix*phiy
    wt[mask] = 0
    wt /= wt.sum()
    
    mean_curvature[i] = np.sum(k_rho * wt)



cmap = cm.seismic
rng_pt = np.abs( np.ma.masked_invalid(mean_curvature) ).max()
norm = plt.Normalize(-rng_pt, rng_pt)
colors = cmap(norm(mean_curvature))
new_mask = np.ma.masked_invalid(mean_curvature).mask
colors[new_mask] = np.array([0,0,0,0])

collection = PatchCollection(circles_colored)
collection.set_facecolors([0,0,0,0])
#collection.set_edgecolors([0,0,0,1])

collection.set_edgecolors(colors)

# Curvature of just the curve
k_curve = k_rho.copy()
k_curve[mask] = 0

fig, ax = plt.subplots()

m = ax.pcolormesh(dx, dy, k_curve, norm=norm, cmap='seismic')
plt.colorbar(m)
#ax.add_collection(collection)
ax.scatter(pos[:,0], pos[:,1], c=mean_curvature, norm=norm, cmap='seismic')

for i, curv in enumerate(mean_curvature):
    pt = pos[i]
    if surf_mask[i] and ~new_mask[i]:
        ax.add_artist(Circle(pt, 1.6, fill=False, edgecolor=colors[i]))

plt.show()



