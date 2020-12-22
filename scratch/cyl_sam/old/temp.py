
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
myorange = myorange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1] 
# Fits a circle, centered at (r,x) = (0,a), radius b, to: x = a + sqrt(b**2 - r**2)
fn_fit = lambda rvals, a, b: a + np.sqrt(b**2 - rvals**2)

## Parameters for circle fitting
# In (a, b); a is x pt of circle center, b is circle radius
param_lb = np.array([-np.inf, 0])
param_ub = np.array([0, np.inf])
bounds = (param_lb, param_ub)
p0 = np.array([-1, 2])


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

    dx, dy = np.gradient(rho_mask)

    pts = []
    for ix in range(xvals.size-1):
        xlo = xvals[ix]
        xhi = xvals[ix+1]

        for iy in range(yvals.size-1):
            ylo = yvals[iy]
            yhi = yvals[iy+1]


            bxcross = xcross[ix, iy]
            bycross = ycross[ix, iy]

            if not (bxcross or bycross):
                continue


            ptx = interp1d(xlo, xhi, rho[ix, iy], rho[ix+1, iy], ys=iso) if bxcross else xlo
            pty = interp1d(ylo, yhi, rho[ix, iy], rho[ix, iy+1], ys=iso) if bycross else ylo

            pts.append(np.array([ptx, pty]))

    # last col of x
    for iy in range(yvals.size-1):
        ylo = yvals[iy]
        yhi = yvals[iy+1]

        if not ycross[-1, iy]:
            continue

        pty = interp1d(ylo, yhi, rho[-1, iy], rho[-1, iy+1], ys=iso)

        pts.append(np.array([xvals[-1], pty]))

    # last col of y
    # last col of x
    for ix in range(xvals.size-1):
        xlo = xvals[ix]
        xhi = xvals[ix+1]

        if not xcross[ix, -1]:
            continue

        ptx = interp1d(xlo, xhi, rho[ix, -1], rho[ix+1, -1], ys=iso)

        pts.append(np.array([ptx, yvals[-1]]))

    return np.array(pts)


ds = np.load('rhoxyz.dat.npz')
xbins = ds['xbins']
ybins = ds['ybins']
zbins = ds['zbins']

xvals = xbins[:-1] + 0.5*np.diff(xbins)
yvals = ybins[:-1] + 0.5*np.diff(ybins)
zvals = zbins[:-1] + 0.5*np.diff(zbins)

rho = ds['rho'].mean(axis=0)

xx, yy = np.meshgrid(xbins, ybins, indexing='ij')
dx = np.diff(xvals)[0]
dy = np.diff(yvals)[0]
dz = np.diff(zvals)[0]


avg_rho = rho / (0.033*dx*dy*dz)
avg_rho = np.clip(avg_rho, 0, 1)
mask_rho = (avg_rho > 0.5).astype(int)


test1 = np.array([[1,0,0],
                  [1,0,0]])[::-1,:].T
test2 = np.array([[1,1,0],
                  [1,0,0]])[::-1,:].T

xbins = np.arange(4)
ybins = np.arange(3)
zbins = np.arange(2)

xvals = xbins[:-1] #+ 0.5*np.diff(xbins)
yvals = ybins[:-1] #+ 0.5*np.diff(ybins)
zvals = zbins[:-1] #+ 0.5*np.diff(zbins)

xx, yy = np.meshgrid(xbins, ybins, indexing='ij')

