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

import numpy as np
import matplotlib.pyplot as plt

from skimage import measure


dat = np.load('rhoxyz.dat.npz')
rho = dat['rho']

n_frames = rho.shape[0]
max_verts = 0

xbins = dat['xbins']
ybins = dat['ybins']
zbins = dat['zbins']

dx = np.diff(xbins)[0]
dy = np.diff(ybins)[0]
dz = np.diff(zbins)[0]

p0 = np.array([xbins[0], ybins[0], zbins[0]])

expt_waters = 0.033 * dx * dy * dz

rho /= expt_waters

for i_frame in range(n_frames):
    this_rho = rho[i_frame]

    verts, faces, normals, values = measure.marching_cubes_lewiner(this_rho, 0.5, spacing=(dx,dy,dz))

    if verts.shape[0] > max_verts:
        max_verts = verts.shape[0]


univ = MDAnalysis.Universe.empty(n_atoms=max_verts, trajectory=True)

with MDAnalysis.Writer("traj_rho.xtc", univ.atoms.n_atoms) as W:
    for i_frame in range(n_frames):

        this_rho = rho[i_frame]

        verts, faces, normals, values = measure.marching_cubes_lewiner(this_rho, 0.5, spacing=(dx,dy,dz))
        n_verts = verts.shape[0]
        univ.atoms[:n_verts].positions = verts + p0

        W.write(univ.atoms)

univ.atoms.write("traj_rho.gro")

avg_rho = rho.mean(axis=0)
verts, faces, normals, values = measure.marching_cubes_lewiner(avg_rho, 0.5, spacing=(dx, dy, dz))
univ = MDAnalysis.Universe.empty(n_atoms=verts.shape[0], trajectory=True)
univ.atoms.positions = verts + p0
univ.atoms.write('traj_avg.gro')



