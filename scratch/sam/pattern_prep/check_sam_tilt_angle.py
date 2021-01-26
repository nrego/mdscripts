
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


def calc_tilt_angle(res):

    atm_s = res.atoms[-1]
    ## Check for CH3!!! - should be pos-res atom
    atm_c = res.atoms[-9]

    vec = atm_c.position - atm_s.position
    vec /= np.linalg.norm(vec)

    ## Find angle between vector and norm to yz plane
    theta = np.arccos(np.dot(vec, np.array([1,0,0]))) * (180/np.pi)

    return theta

## Long trajectory, calculate angle between SAM alkyl (vector from S to anchored C)
##.   and the normal axis to the plane. should be about 28 deg.
univ = MDAnalysis.Universe("confout.gro", "traj.xtc")

residues = univ.select_atoms("resname OH or resname CH3").residues

n_res = residues.n_residues

avg_angle = np.zeros((univ.trajectory.n_frames, n_res))

for i_frame, ts in enumerate(univ.trajectory):

    if i_frame % 100 == 0:
        print("Doing frame {} of {}".format(i_frame+1, univ.trajectory.n_frames))

    for i_res, res in enumerate(residues):
        angle = calc_tilt_angle(res)
        avg_angle[i_frame, i_res] = angle



