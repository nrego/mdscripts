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

import math

gaus = lambda r, sig_sq: (1/(np.sqrt(2*np.pi*sig_sq))) * np.exp(-r**2/(2*sig_sq))

# Find charge density of a group of atoms (ag, AtomGroup)
##  according to their point charges and positions in 
###   grid space
def charge_density(bx, by, bz, xx, yy, zz, d, dcg, ag):

    for atm in ag:
        charge = atm.charge

        pos = atm.position
        idx_x = np.digitize(pos[0], bx)-1
        idx_y = np.digitize(pos[1], by)-1
        idx_z = np.digitize(pos[2], bz)-1

        d[idx_x, idx_y, idx_z] += charge

        phix = gaus(pos[0]-xx, 2)
        phiy = gaus(pos[1]-yy, 2)
        phiz = gaus(pos[2]-zz, 2)

        phi = phix*phiy*phiz

        dcg += charge*phi





