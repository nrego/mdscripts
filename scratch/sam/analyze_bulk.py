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

import numpy as np

from scratch.sam.util import *


def extract_probe_vol(umbr_path):
    with open(umbr_path, 'r') as fin:
        lines = fin.readlines()

    min_x, min_y, min_z, max_x, max_y, max_z = [float(s) for s in lines[3].split()]

    return (np.round(max_x-min_x, 4), np.round(max_y-min_y, 4), np.round(max_z-min_z, 4))


fnames = sorted(glob.glob('P*/PvN.dat'))

n_dat = len(fnames)

energies = np.zeros(n_dat)
errs = np.zeros_like(energies)
vols = np.zeros_like(energies)
sas = np.zeros_like(energies)

myfeat = np.zeros((n_dat, 3))

for i, fname in enumerate(fnames):
    dirname = os.path.dirname(fname)

    dx, dy, dz = extract_probe_vol('{}/umbr.conf'.format(dirname))
    dat = np.loadtxt(fname)
    this_ener = dat[0,1]
    this_err  = dat[0,2]

    energies[i] = this_ener
    errs[i] = this_err
    vols[i] = dx*dy*dz
    sas[i] = dy*dz + dx*(dy+dz)

    myfeat[i] = dy*dz, dy, dz

