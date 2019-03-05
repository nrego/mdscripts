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

def plot_3d(x, y, z, colors='k'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, c=colors)

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
pos = np.loadtxt('pos.dat')

buff = 4
res = 1

min_pt = np.floor(pos.min()) - buff
max_pt = np.ceil(pos.max()) + buff

grid_pts = np.arange(min_pt, max_pt+res, res, dtype=np.float32)

YY, XX, ZZ = np.meshgrid(grid_pts, grid_pts, grid_pts)

rho = np.zeros_like(XX)
sig = 1
rcut = 10


for i in range(pos.shape[0]):
#for i in range(1):
    pt = pos[i]
    phix = phi(XX-pt[0], sig, sig**2, rcut, rcut**2)
    phiy = phi(YY-pt[1], sig, sig**2, rcut, rcut**2)
    phiz = phi(ZZ-pt[2], sig, sig**2, rcut, rcut**2)

    rho += phix*phiy*phiz


rho /= (40.0 / 1000)

verts, faces, normals, values = measure.marching_cubes_lewiner(rho, 0.5, spacing=(1,1,1))

mesh = Poly3DCollection(verts[faces]+min_pt)
mesh.set_edgecolor('k')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.add_collection(mesh)
#ax.plot(verts[:,0]+min_pt, verts[:,1]+min_pt, verts[:,2]+min_pt)
ax.set_xlim(min_pt, max_pt)
ax.set_ylim(min_pt, max_pt)
ax.set_zlim(min_pt, max_pt)
#ax.scatter(pos[:,0], pos[:,1], pos[:,2], 'k')

fileout = 'blah.dx'

rho_shape = np.reshape(rho, rho.size)
with open(fileout, 'w') as f:
    f.write("object 1 class gridpositions counts {} {} {}\n".format(*rho.shape))
    f.write("origin {:1.8e} {:1.8e} {:1.8e}\n".format(min_pt, min_pt, min_pt))
    f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(res,0,0))
    f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0,res,0))
    f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0,0,res))
    f.write("object 2 class gridconnections counts {} {} {}\n".format(*rho.shape))
    f.write("object 3 class array type double rank 0 items {} data follows\n".format(rho_shape.size))
    cntr = 0 
    for pt in rho_shape:
        f.write("{:1.8e} ".format(pt))
        cntr += 1
        if (cntr % 3 ==0):
            f.write("\n")

