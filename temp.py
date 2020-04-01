from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
import logging

import MDAnalysis
#from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

from IPython import embed

from scipy.spatial import cKDTree
import itertools
#from skimage import measure

from rhoutils import rho, cartesian
from mdtools import ParallelTool

from constants import SEL_SPEC_HEAVIES, SEL_SPEC_HEAVIES_NOWALL
from mdtools.fieldwriter import RhoField
import sys
import argparse

parser = argparse.ArgumentParser('Output cavity voxels for each frame')
parser.add_argument('-c', '--top', type=str, default='ofile.gro', help='input structure file')
parser.add_argument('-f', '--traj', type=str, default='ofile.xtc', help='Input trajectory')
parser.add_argument('-b', '--start', default=500, type=int, help='start time, in ps')
args = parser.parse_args()

xmin = 28.0
ymin = 15.0
zmin = 15.0

xmax = 38.5
ymax = 55.0
zmax = 55.0

ycom = ymin + (ymax - ymin)/2.0
zcom = zmin + (zmax - zmin)/2.0

dx = 0.4
xvals = np.arange(xmin, xmax+dx, dx)
dr = 0.2
rvals = np.arange(0, 20+dr, dr)

univ = MDAnalysis.Universe(args.top, args.traj)
start_frame = int(args.start / univ.trajectory.dt)
n_frames = univ.trajectory.n_frames

n_waters = np.zeros(n_frames-start_frame)
water_com = np.zeros((n_frames-start_frame, 3))

for i, i_frame, in enumerate(np.arange(start_frame, n_frames)):
    if i_frame % 100 == 0:
        print("frame: {}".format(i_frame))
        sys.stdout.flush()

    univ.trajectory[i_frame]
    waters = univ.select_atoms("name OW")
    water_pos = waters.positions

    selx = (water_pos[:,0] >= xmin) & (water_pos[:,0] < xmax)
    sely = (water_pos[:,1] >= ymin) & (water_pos[:,1] < ymax)
    selz = (water_pos[:,2] >= zmin) & (water_pos[:,2] < zmax)

    sel_mask = selx & sely & selz

    this_waters = waters[sel_mask]
    water_com[i] = this_waters.positions.mean(axis=0)

    n_waters[i] = sel_mask.sum()

np.savetxt("phiout_cube.dat", n_waters, fmt='%3d')
np.save("com_noshift.dat", water_com)
