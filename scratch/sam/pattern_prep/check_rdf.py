
import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

import time

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from scratch.sam.util import *

from scratch.neural_net.lib import *

from scipy.spatial import cKDTree


univ = MDAnalysis.Universe("confout.gro", "traj.xtc")

n_frames = univ.trajectory.n_frames

min_pt = np.array([8.2, 14.25])
max_pt = np.array([31.9, 43.75])

min_x, min_y = min_pt
max_x, max_y = max_pt

# thickness of slabs, in A
dz = 0.5
zbounds = np.arange(0, 85+dz, dz)
zc = zbounds[:-1] + 0.5*dz

n_of_z = np.zeros((n_frames, zc.size))
sam_of_z = np.zeros_like(n_of_z)


waters = univ.select_atoms("name OW")
sam = univ.select_atoms("(resname OH or resname CH3) and not name H*")

for i_frame in range(n_frames):
    univ.trajectory[i_frame]
    if i_frame % 100 == 0:
        print("Doing frame {} of {}".format(i_frame+1, n_frames))
    for i in range(zc.size):
        lb = zbounds[i]
        ub = zbounds[i+1]

        z_mask = (waters.positions[:,2] >= lb) & (waters.positions[:,2] < ub)
        x_mask = (waters.positions[:,0] >= 8.2) & (waters.positions[:,0] < 31.9)
        y_mask = (waters.positions[:,1] >= 14.25) & (waters.positions[:,1] < 43.75)

        mask = (x_mask & y_mask & z_mask)
        sel = waters[mask]

        n_of_z[i_frame, i] = sel.n_atoms

        z_mask = (sam.positions[:,2] >= lb) & (sam.positions[:,2] < ub)
        x_mask = (sam.positions[:,0] >= 8.2) & (sam.positions[:,0] < 31.9)
        y_mask = (sam.positions[:,1] >= 14.25) & (sam.positions[:,1] < 43.75)

        mask = (x_mask & y_mask & z_mask)
        sel = sam[mask]

        sam_of_z[i_frame, i] = sel.n_atoms

avg_n_of_z = n_of_z.mean(axis=0)
avg_sam_of_z = sam_of_z.mean(axis=0)

dat = np.vstack((zc, avg_n_of_z, avg_sam_of_z)).T

np.savetxt("rdf.dat", dat)



