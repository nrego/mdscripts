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


plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})

def get_min_err(dsname):
    min_idx = dsname['energies'].argmin()
    assert dsname['states'][min_idx].k_o == 0

    return dsname['energies'][min_idx], dsname['err_energies'][min_idx]

def get_max_err(dsname):
    max_idx = dsname['energies'].argmax()
    assert dsname['states'][max_idx].k_c == 0

    return dsname['energies'][max_idx], dsname['err_energies'][max_idx]

def extract_probe_vol(umbr_path):
    with open(umbr_path, 'r') as fin:
        lines = fin.readlines()

    min_x, min_y, min_z, max_x, max_y, max_z = [float(s) for s in lines[3].split()]

    return (np.round(max_x-min_x, 4), np.round(max_y-min_y, 4), np.round(max_z-min_z, 4))

### Extract all pure methyl and hydroxyl free energies
#########################################

## Extract all f_0's
headdirs = sorted(glob.glob('P*'))

energies = []
errs = []
feat_vec = []
dx = []
dy = []
dz = []

for headdir in headdirs:

    pathnames = list(reversed(sorted( glob.glob('{}/k_*/PvN.dat'.format(headdir)) )))

    if len(pathnames) == 0:
        continue

    size_list = np.array(headdir[1:].split('_'), dtype=int)

    try:
        P, Q = size_list
    except ValueError:
        P, Q = size_list[0], size_list[0]

    this_dx, this_dy, this_dz = extract_probe_vol('{}/umbr.conf'.format(headdir))


    for pathname in pathnames:
        print("doing {}".format(pathname))
        kc = int(pathname.split('/')[1].split('_')[1])
        if kc != P*Q and kc != 0:
            continue
        pvn = np.loadtxt(pathname)

        if kc == P*Q:
            pt_idx = np.arange(P*Q, dtype=int)
        else:
            pt_idx = np.array([], dtype=int)

        state = State(pt_idx, ny=P, nz=Q)
        ko = state.k_o
        n_oo = state.n_oo
        n_oe = state.n_oe
        energies.append(pvn[0,1])
        errs.append(pvn[0,2])
        feat_vec.append([P, Q, ko, n_oo, n_oe])
        dx.append(this_dx)
        dy.append(this_dy)
        dz.append(this_dz)

# 6 x 6 meth
e, err = get_min_err(np.load('sam_pattern_06_06.npz'))
feat_vec.append([6, 6, 0, 0, 0])
energies.append(e)
errs.append(err)

# 6 x 6 hydroxyl
e, err = get_max_err(np.load('sam_pattern_06_06.npz'))
feat_vec.append([6, 6, 36, 85, 46])
energies.append(e)
errs.append(err)

this_dx, this_dy, this_dz = extract_probe_vol('../pattern_sample/umbr.conf')
dx.append(this_dx)
dy.append(this_dy)
dz.append(this_dz)

dx.append(this_dx)
dy.append(this_dy)
dz.append(this_dz)

# 4 x 9 meth
e, err = get_min_err(np.load('sam_pattern_04_09.npz'))
feat_vec.append([4, 9, 0, 0, 0])
energies.append(e)
errs.append(err)

# 4 x 9 hydroxyl
e, err = get_max_err(np.load('sam_pattern_04_09.npz'))
feat_vec.append([4, 9, 36, 83, 50])
energies.append(e)
errs.append(err)

this_dx, this_dy, this_dz = extract_probe_vol('P4_9/umbr.conf')
dx.append(this_dx)
dy.append(this_dy)
dz.append(this_dz)

dx.append(this_dx)
dy.append(this_dy)
dz.append(this_dz)

# 4 x 4 meth
e, err = get_min_err(np.load('sam_pattern_04_04.npz'))
feat_vec.append([4, 4, 0, 0, 0])
energies.append(e)
errs.append(err)

# 4 x 4 hydroxyl
e, err = get_max_err(np.load('sam_pattern_04_04.npz'))
feat_vec.append([4, 4, 16, 33, 30])
energies.append(e)
errs.append(err)

this_dx, this_dy, this_dz = extract_probe_vol('P4/umbr.conf')
dx.append(this_dx)
dy.append(this_dy)
dz.append(this_dz)

dx.append(this_dx)
dy.append(this_dy)
dz.append(this_dz)


energies = np.array(energies)
feat_vec = np.array(feat_vec)
errs = np.array(errs)
dx = np.array(dx)
dy = np.array(dy)
dz = np.array(dz)

np.savez_compressed('sam_pattern_pure', energies=energies, feat_vec=feat_vec, err_energies=errs, 
                    dx=dx, dy=dy, dz=dz, header='featvec:  P  Q  k_o  n_oo  n_oe')


