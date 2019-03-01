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
from matplotlib.collections import PatchCollection

import MDAnalysis

import os, glob
import cPickle as pickle
from scipy.integrate import cumtrapz

import math

from rhoutils import phi_1d

from skimage import measure

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
pos = np.loadtxt('{}/pos.dat'.format(homedir), dtype=np.float32)
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
#ax.add_collection(collection)

#plt.scatter(pos[:,0], pos[:,1], color='k')

