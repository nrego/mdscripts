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
#ax.add_collection(mesh)
ax.scatter(verts[:,0]+min_pt, verts[:,1]+min_pt, verts[:,2]+min_pt)
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



# X, Y, Z all 2d arrays (e.g. X(u,v), etc)
def find_surf_curvature(X, Y, Z):
    # First derivatives
    Xu, Xv = np.gradient(X)
    Yu, Yv = np.gradient(Y)
    Zu, Zv = np.gradient(Z)

    # Second derivatives
    Xuu, Xuv = np.gradient(Xu)
    Yuu, Yuv = np.gradient(Yu)
    Zuu, Zuv = np.gradient(Zu)

    Xvu, Xvv = np.gradient(Xv)
    Yvu, Yvv = np.gradient(Yv)
    Zvu, Zvv = np.gradient(Zv)

    # 'sig(u,v)' is the parametric representation of the surface 
    #      These are its derivatives for each (u,v) gridpoint 
    sig_u = np.vstack((Xu.ravel(), Yu.ravel(), Zu.ravel())).T
    sig_v = np.vstack((Xv.ravel(), Yv.ravel(), Zv.ravel())).T

    sig_uu = np.vstack((Xuu.ravel(), Yuu.ravel(), Zuu.ravel())).T
    sig_uv = np.vstack((Xuv.ravel(), Yuv.ravel(), Zuv.ravel())).T
    sig_vv = np.vstack((Xvv.ravel(), Yvv.ravel(), Zvv.ravel())).T

    # First fundamental coefficients 
    E = (sig_u**2).sum(axis=1)
    F = (sig_u * sig_v).sum(axis=1)
    G = (sig_v * sig_v).sum(axis=1)

    m = np.cross(sig_u, sig_v)
    p = np.sqrt(np.sum(m**2, axis=1))
    # normals
    n = m/p[:,None]

    # Second fundamental coefficients
    L = (sig_uu * n).sum(axis=1)
    M = (sig_uv * n).sum(axis=1)
    N = (sig_vv * n).sum(axis=1)

    nu, nv = X.shape

    # Gaussian curvature for each sig(u,v) pt
    K = (L*N - M**2) / (E*G - F**2)
    K = K.reshape(nu, nv)

    # Mean curvature for each sig(u,v) pt
    H = (E*N + G*L - 2*F*M) / (2*(E*G - F**2))
    H = H.reshape(nu, nv)

    # principal curvatures
    kap_max = H + np.sqrt(H**2 - K)
    kap_min = H - np.sqrt(H**2 - K)


    return (K, H, kap_min, kap_max)


## Sample stuff with torus - parametric surface ## 


# Generate torus mesh
angle = np.linspace(0, 2 * np.pi, 32)
theta, phi = np.meshgrid(angle, angle)
r, R = .25, 1.
X = (R + r * np.cos(phi)) * np.cos(theta)
Y = (R + r * np.cos(phi)) * np.sin(theta)
Z = r * np.sin(phi)

# Display the mesh
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

K, H, kap_min, kap_max = find_surf_curvature(X, Y, Z)

## plot and color according to gaussian curvature ##
rng_pt = np.abs(kap_max.max() * kap_min.min())
norm = plt.Normalize(-rng_pt, rng_pt)
colors = cm.seismic(norm(K))
ax.plot_surface(X, Y, Z, facecolors=colors, rstride = 1, cstride = 1)
plt.show()


