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

dx, dy = np.meshgrid(grid_pts, grid_pts)

rho = np.zeros_like(dx)
sig = 1
rcut = 10

circles = []
for i in range(pos.shape[0]):
    pt = pos[i]
    if surf_mask[i]:
        circles.append(Circle(pt, 1.6, fill=False))
    phix = phi(dx-pt[0], sig, sig**2, rcut, rcut**2)
    phiy = phi(dy-pt[1], sig, sig**2, rcut, rcut**2)

    rho += phix*phiy

cmap = cm.hot_r
norm = mpl.colors.Normalize(0, rho.max())

fig, ax = plt.subplots(figsize=(6,6))
collection = PatchCollection(circles)
collection.set_facecolors([0,0,0,0])
collection.set_edgecolors([0,0,0,1])

ax.pcolormesh(dx, dy, rho, cmap=cmap, norm=norm)

contour = measure.find_contours(rho, 0.5)[0]
contour *= res
contour += min_pt
contour[:,0], contour[:,1] = contour[:,1], contour[:,0].copy()
#plt.plot(contour[:,1], contour[:,0])
points = contour.reshape(-1,1,2)

segments = np.concatenate([points[:-1], points[1:]], axis=1)

cmap2 = cm.seismic
norm = plt.Normalize(0, segments.shape[0])
lc = LineCollection(segments, cmap='rainbow', norm=norm)
lc.set_array(np.arange(segments.shape[0]))

ax.add_collection(lc)

plt.show()


## Find curvature from finite differences

dx_ds = np.gradient(contour[:,0])
dy_ds = np.gradient(contour[:,1])

d2x_ds2 = np.gradient(dx_ds)
d2y_ds2 = np.gradient(dy_ds)

curvature = (dx_ds*d2y_ds2 - dy_ds*d2x_ds2) / (dx_ds**2 + dy_ds**2)**1.5
range_pt = np.max(np.abs(curvature))

norm = plt.Normalize(-range_pt, range_pt)

plt.plot(contour[:,0], contour[:,1], 'k-')
plt.scatter(contour[:,0], contour[:,1], c=curvature, cmap='seismic', norm=norm)

