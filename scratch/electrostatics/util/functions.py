from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import networkx as nx
from scipy.spatial import cKDTree

from scipy.integrate import cumtrapz

import math, scipy

from rhoutils import rho2, gaus, gaus_1d, interp1d

#gaus = lambda r, sig_sq: (1/(np.sqrt(2*np.pi*sig_sq))) * np.exp(-r**2/(2*sig_sq))
sig = 4.5/np.sqrt(2)
#sig = 1
sig_sq = sig**2
sqrt_sig = np.sqrt(2)*sig


# Find charge density of a group of atoms (ag, AtomGroup)
##  according to their point charges and positions in 
###   grid space
def charge_density(grid_x, grid_y, grid_z, xx, yy, zz, ag):

    x_ravel = xx.ravel()
    y_ravel = yy.ravel()
    z_ravel = zz.ravel()

    # All grid points, in a (n_pts, 3) array
    #pts = np.vstack((x_ravel, y_ravel, z_ravel)).T
    y_ravel2 = yy[20,...].ravel()
    z_ravel2 = zz[20,...].ravel()
    x_pt = 20.

    pts = np.vstack((np.ones_like(y_ravel2)*x_pt, y_ravel2, z_ravel2)).T
    # Point charge density, mesh
    d = np.zeros_like(xx)
    # Smoothed charge density, unraveled
    dcg = np.zeros_like(x_ravel)

    dcg = np.zeros_like(grid_x)

    potential = np.zeros(pts.shape[0])
    for i, atm in enumerate(ag):
        #if i % 1000 == 0:
        #    print(i)
        charge = atm.charge

        pos = atm.position
        idx_x = np.digitize(pos[0], grid_x)-1
        idx_y = np.digitize(pos[1], grid_y)-1
        idx_z = np.digitize(pos[2], grid_z)-1

        d[idx_x, idx_y, idx_z] += charge

        #phi = gaus(pos-pts, sig, sig_sq)
        #dcg += charge*phi
        dcg += charge*gaus_1d(pos[0]-grid_x, sig, sig_sq)
        r = np.sqrt(((pos-pts)**2).mean(axis=1))
        potential += (charge/r) * scipy.special.erfc(r/(sqrt_sig))

    potential = potential.reshape((d.shape[1], d.shape[2]))

    return d, dcg, potential




