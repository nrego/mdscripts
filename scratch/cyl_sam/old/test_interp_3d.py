
import sys
import numpy as np
from math import sqrt
import argparse
import logging
import shutil
import MDAnalysis

from IPython import embed

from scipy.spatial import cKDTree
import itertools
#from skimage import measure

from rhoutils import rho, cartesian
from mdtools import ParallelTool

from constants import SEL_SPEC_HEAVIES, SEL_SPEC_HEAVIES_NOWALL
from mdtools.fieldwriter import RhoField
import sys
import argparse, os
from scipy.interpolate import interp2d

import matplotlib as mpl

from skimage import measure
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

homedir = os.environ['HOME']

# Find xs, point where line between (xlo, ylo) and (xhi, yhi) crosses ys
def interp1d(xlo, xhi, ylo, yhi, ys=0.5):

    m = (yhi - ylo) / (xhi - xlo)

    if m == 0:
        return xlo

    return xlo + (ys-ylo)/m


# Given a density field (shape: (xvals.size, yvals.size, zvals.size)), find 
#     linearly interpolated points where we cross isovalue
def get_interp_points(rho, xvals, yvals, zvals, iso=0.5):


    rho_mask = (rho > iso).astype(int)
    xcross = np.diff(rho_mask, axis=0).astype(bool)
    ycross = np.diff(rho_mask, axis=1).astype(bool)
    zcross = np.diff(rho_mask, axis=2).astype(bool)

    dx, dy, dz = np.gradient(rho_mask)

    pts = []
    for ix in range(xvals.size-1):
        xlo = xvals[ix]
        xhi = xvals[ix+1]

        for iy in range(yvals.size-1):
            ylo = yvals[iy]
            yhi = yvals[iy+1]

            for iz in range(zvals.size-1):
                zlo = zvals[iz]
                zhi = zvals[iz+1]

                bxcross = xcross[ix, iy, iz]
                bycross = ycross[ix, iy, iz]
                bzcross = zcross[ix, iy, iz]

                if not (bxcross or bycross or bzcross):
                    continue

                ptx = interp1d(xlo, xhi, rho[ix, iy, iz], rho[ix+1, iy, iz], ys=iso) if bxcross else xlo
                pty = interp1d(ylo, yhi, rho[ix, iy, iz], rho[ix, iy+1, iz], ys=iso) if bycross else ylo
                ptz = interp1d(zlo, zhi, rho[ix, iy, iz], rho[ix, iy, iz+1], ys=iso) if bzcross else zlo

                pts.append(np.array([ptx, pty, ptz]))

    # last col of x
    for iy in range(yvals.size-1):
        ylo = yvals[iy]
        yhi = yvals[iy+1]

        for iz in range(zvals.size-1):
            zlo = zvals[iz]
            zhi = zvals[iz+1]

            bycross = ycross[-1, iy, iz]
            bzcross = zcross[-1, iy, iz]

            if not (bycross or bzcross):
                continue

            pty = interp1d(ylo, yhi, rho[-1, iy, iz], rho[-1, iy+1, iz], ys=iso) if bycross else ylo
            ptz = interp1d(zlo, zhi, rho[-1, iy, iz], rho[-1, iy, iz+1], ys=iso) if bzcross else zlo

            pts.append(np.array([xvals[-1], pty, ptz]))

    # last col of y
    for ix in range(xvals.size-1):
        xlo = xvals[ix]
        xhi = xvals[ix+1]

        for iz in range(zvals.size-1):
            zlo = zvals[iz]
            zhi = zvals[iz+1]

            bxcross = xcross[ix, -1, iz]
            bzcross = zcross[ix, -1, iz]

            if not (bxcross or bzcross):
                continue

            ptx = interp1d(xlo, xhi, rho[ix, -1, iz], rho[ix+1, -1, iz], ys=iso) if bxcross else xlo
            ptz = interp1d(zlo, zhi, rho[ix, -1, iz], rho[ix, -1, iz+1], ys=iso) if bzcross else zlo

            pts.append(np.array([ptx, yvals[-1], ptz]))

    # last col of z
    for ix in range(xvals.size-1):
        xlo = xvals[ix]
        xhi = xvals[ix+1]

        for iy in range(yvals.size-1):
            ylo = yvals[iy]
            yhi = yvals[iy+1]

            bxcross = xcross[ix, iy, -1]
            bycross = ycross[ix, iy, -1]

            if not (bxcross or bycross):
                continue

            ptx = interp1d(xlo, xhi, rho[ix, iy, -1], rho[ix+1, iy, -1], ys=iso) if bxcross else xlo
            pty = interp1d(ylo, yhi, rho[ix, iy, -1], rho[ix, iy+1, -1], ys=iso) if bycross else ylo

            pts.append(np.array([ptx, pty, zvals[-1]]))


    return np.array(pts)



ds = np.load('rho_bphi.dat.npz')
#ds = np.load('rhoxyz_dx_10.dat.npz')
xbins = ds['xbins']
ybins = ds['ybins']
zbins = ds['zbins']

xvals = xbins[:-1] + 0.5*np.diff(xbins)
yvals = ybins[:-1] + 0.5*np.diff(ybins)
zvals = zbins[:-1] + 0.5*np.diff(zbins)

#rho = ds['rho'].mean(axis=0)
rho0 = ds['rho0']
rho_bphi = ds['rho_bphi']
beta_phi_vals = ds['beta_phi_vals']

xx, yy = np.meshgrid(xbins, ybins, indexing='ij')
dx = np.diff(xvals)[0]
dy = np.diff(yvals)[0]
dz = np.diff(zvals)[0]

max_atms = -np.inf
for i, bphi in enumerate(beta_phi_vals):

    rho = rho_bphi[i]
    #avg_rho = rho / (0.033*dx*dy*dz)
    avg_rho = rho / rho0
    #avg_rho = np.clip(avg_rho, 0, 1)
    mask_rho = (avg_rho > 0.5).astype(int)

    pts = get_interp_points(avg_rho, xvals, yvals, zvals)

    #plt.close()
    #ax = plt.gca(projection='3d')
    #ax.scatter(pts[:,0], pts[:,1], pts[:,2])

    univ = MDAnalysis.Universe.empty(n_atoms=pts.shape[0], trajectory=True)
    if univ.atoms.n_atoms > max_atms:
        max_atms = univ.atoms.n_atoms
    univ.atoms.positions = pts

    #univ.atoms.write("{}/Desktop/avg_inter_{:04d}.gro".format(homedir, int(bphi*100)))

univ = MDAnalysis.Universe.empty(n_atoms=max_atms, trajectory=True)

with MDAnalysis.Writer("traj.xtc", univ.atoms.n_atoms) as W:

    for i, bphi in enumerate(beta_phi_vals):
        univ.atoms.positions[:] = 0
        rho = rho_bphi[i]
        avg_rho = rho / rho0

        mask_rho = (avg_rho > 0.5).astype(int)

        pts = get_interp_points(avg_rho, xvals, yvals, zvals)
        blah = univ.atoms.positions.copy()
        blah[:pts.shape[0]] = pts
        univ.atoms.positions = blah

        W.write(univ.atoms)

        if i == 0:
            univ.atoms.write("base.gro")

