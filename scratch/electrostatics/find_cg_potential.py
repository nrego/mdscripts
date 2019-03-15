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

#from util import *

univ = MDAnalysis.Universe('top.tpr', 'whole.xtc')
ag = univ.select_atoms('resname SOL')

box = np.array([85., 52., 60.])

res = cd
grid_x = np.arange(0, box[0]+res, res)
grid_y = np.arange(0, box[1]+res, res)
grid_z = np.arange(0, box[2]+res, res)

xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

densities = np.zeros((univ.trajectory.n_frames,  xx.shape[0], xx.shape[1], xx.shape[2]))
d_cg = np.zeros_like(densities)

for i, ts in enumerate(univ.trajectory):
    this_d = np.zeros_like(xx)
    this_dcg = np.zeros_like(xx)

    print("frame: {}".format(i))

    charge_density(grid_x, grid_y, grid_z, xx, yy, zz, this_d, this_dcg, ag)

    densities[i,:] = this_d
    d_cg[i,:] = this_dcg

