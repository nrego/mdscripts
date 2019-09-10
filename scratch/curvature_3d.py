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
import pickle
from scipy.integrate import cumtrapz

import math

from rhoutils import phi_1d

from skimage import measure

from scipy.interpolate import CubicSpline

import mdtraj as md

## Testing ##
gaus = lambda x_sq, sig_sq: np.exp(-x_sq/(2*sig_sq))

def write_pdb(fileout, pts, vals=None):
    n_atoms = pts.shape[0]
    if vals is None:
        vals = np.zeros(n_atoms)
    top = md.Topology()
    c = top.add_chain()

    cnt = 0
    for i in range(n_atoms):
        cnt += 1
        r = top.add_residue('II', c)
        a = top.add_atom('II', md.element.get_by_symbol('VS'), r, i)

    with md.formats.PDBTrajectoryFile(fileout, 'w') as f:
        # Mesh pts have to be in nm
        f.write(pts, top)

    univ = MDAnalysis.Universe(fileout)
    univ.add_TopologyAttr('tempfactors')
    univ.atoms.tempfactors = vals
    univ.atoms.write(fileout, bonds=None)


def dump_dx(fileout, rho_shape, res, min_pt):
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

buff = 2
res = 0.5
sig = 1
rcut = 5

#pos = np.loadtxt('pos.dat')
univ = MDAnalysis.Universe("bphi.pdb")
pos = univ.atoms.positions
min_pt = pos.min()-buff
max_pt = pos.max()+buff

grid_pts = np.arange(min_pt, max_pt+res, res)


XX, YY, ZZ = np.meshgrid(grid_pts, grid_pts, grid_pts, indexing='ij')

## First find coarse-grained protein density so we can construct an isosurface
#     over the protein

# atoms per A^3
density = 0.040

rho = np.zeros_like(XX)

for i, pt in enumerate(pos):
    phix = phi(pt[0]-XX, sig, sig**2, rcut, rcut**2)
    phiy = phi(pt[1]-YY, sig, sig**2, rcut, rcut**2)
    phiz = phi(pt[2]-ZZ, sig, sig**2, rcut, rcut**2)

    rho += phix*phiy*phiz

rho /= density

# Save rho as dx file
print("...saving protein density field...")
dump_dx('prot_isosurf.dx', rho.ravel(), res, min_pt)

# surface threshold
s = 0.5
verts, faces, normals, values = measure.marching_cubes_lewiner(rho, s, spacing=(1,1,1))
verts = verts*res + min_pt

#ax = plt.gca(projection='3d')
#ax.scatter(verts[:,0], verts[:,1], verts[:,2])
#plt.show()


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

# Now save the curvature as a 'density' field
dump_dx('prot_curv.dx', k_mean_rho.ravel(), res, min_pt)

## Plot slice where z = (z_max-z_min)/2
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

# grid points on the isosurface
pts_all = np.vstack([XX[isosurf_mask].ravel(), YY[isosurf_mask].ravel(), ZZ[isosurf_mask].ravel()]).T

norm = plt.Normalize(-1,1)

ax = plt.gca(projection='3d')
ax.scatter(pts_all[:,0], pts_all[:,1], pts_all[:,2], c=isosurf_curv_vals, norm=norm, cmap='seismic')
plt.show()

write_pdb('surf.pdb', pts_all, isosurf_curv_vals)

## Color atoms by nearby curvature ##
#univ = MDAnalysis.Universe('actual_contact.pdb')
univ.atoms.tempfactors = 0
ax = plt.gca(projection='3d')
rcut = 5
sig = 1
for i, pt in enumerate(pos):
    phix = phi(pt[0]-XX, sig, sig**2, rcut, rcut**2)
    phiy = phi(pt[1]-YY, sig, sig**2, rcut, rcut**2)
    phiz = phi(pt[2]-ZZ, sig, sig**2, rcut, rcut**2)

    wt = phix*phiy*phiz
    wt[~isosurf_mask] = 0
    wt /= wt.sum()

    atm_curv = np.sum(wt*isosurf_curvature_mean)

    univ.atoms[i].tempfactor = atm_curv

univ.atoms.write('curv.pdb', bonds=None)

np.savetxt('atomic_curvature.dat', univ.atoms.tempfactors)

vert_curv = []
for vert in verts:
    phix = phi(vert[0]-XX, sig, sig**2, rcut, rcut**2)
    phiy = phi(vert[1]-YY, sig, sig**2, rcut, rcut**2)
    phiz = phi(vert[2]-ZZ, sig, sig**2, rcut, rcut**2)

    wt = phix*phiy*phiz
    wt[~isosurf_mask] = 0
    wt /= wt.sum()

    this_curv = np.sum(wt*isosurf_curvature_mean)

    vert_curv.append(this_curv)   


