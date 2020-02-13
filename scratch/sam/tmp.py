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

    return (min_x, np.round(max_x-min_x, 4), np.round(max_y-min_y, 4), np.round(max_z-min_z, 4))


pvn_bulk = np.loadtxt('PvN.dat')
dg_bulk = pvn_bulk[0,1]
err_dg_bulk = pvn_bulk[0,2]
nvphi_bulk = np.loadtxt('NvPhi.dat')
n0_bulk = nvphi_bulk[0,1]
min_x_bulk, dx_bulk, dy_bulk, dz_bulk = extract_probe_vol('umbr.conf')
v_bulk = np.prod(np.array([dx_bulk, dy_bulk, dz_bulk]))


print('\n############')
print('Bulk vol: {:0.2f} nm^3'.format(v_bulk))
print('  dx: {:0.2f}  dy: {:0.2f}  dz: {:0.2f}'.format(dx_bulk, dy_bulk, dz_bulk))
print('  min_x: {:0.4f}'.format(min_x_bulk))
print('  Expt number waters: {:.2f}'.format(v_bulk*33.0))
print('  Actual avg: {:0.2f}'.format(n0_bulk))
print('\n  beta Pv(0): {:0.2f} ({:0.2f})'.format(dg_bulk, err_dg_bulk))

dnames = sorted(glob.glob('surf*'))

for dirname in dnames:
    try:
        pvn = np.loadtxt('{}/PvN.dat'.format(dirname))
        nvphi = np.loadtxt('{}/NvPhi.dat'.format(dirname))
    except:
        continue

    dg = pvn[0,1]
    err_dg =pvn[0,2]
    n0 = nvphi[0,1]
    min_x, dx, dy, dz = extract_probe_vol('{}/umbr.conf'.format(dirname))
    this_v = np.prod(np.array([dx, dy, dz]))

    univ = MDAnalysis.Universe('{}/nstar_108/confout.gro'.format(dirname))
    min_x0 = univ.select_atoms('name OW').positions[:,0].min() / 10
    univ = MDAnalysis.Universe('{}/nstar_neg_044/confout.gro'.format(dirname))
    min_x20 = univ.select_atoms('name OW').positions[:,0].min() / 10

    print('\n############')
    print(dirname)
    print('vol: {:0.2f} nm^3'.format(this_v))
    print('  dx: {:0.2f}  dy: {:0.2f}  dz: {:0.2f}'.format(dx, dy, dz))
    print('  min_x: {:0.4f}'.format(min_x))
    print('  phi=0 min x: {:0.4f}'.format(min_x0))
    print('  phi=20 min x: {:0.4f}'.format(min_x20))
    print('  Expt number waters: {:.2f}'.format(this_v*33.0))
    print('  Actual avg: {:0.2f}'.format(n0))
    print('\n  beta Pv(0): {:0.2f} ({:0.2f})'.format(dg, err_dg))



