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

### Extract all pure methyl and hydroxyl free energies
#########################################

## Extract all f_0's
headdirs = sorted(glob.glob('P*'))

energies = []
errs = []
feat_vec = []

for headdir in headdirs:

    pathnames = sorted( glob.glob('{}/k_*/PvN.dat'.format(headdir)) )

    if len(pathnames) == 0:
        continue

    size_list = np.array(headdir[1:].split('_'), dtype=int)

    try:
        P, Q = size_list
    except ValueError:
        P, Q = size_list[0], size_list[0]

    for pathname in pathnames:
        kc = int(pathname.split('/')[1].split('_')[1])
        if kc != P*Q or kc != 0:
            continue
        pvn = np.loadtxt(pathname)

        if kc == P*Q:
            pt_idx = np.arange(P*Q, dtype=int)
        else:
            pt_idx = np.array([], dtype=int)

        state = State(pt_idx, ny=P, nz=Q)
        ko = state.ko
        n_oo = state.n_oo
        n_oe = state.n_oe
        energies.append(pvn[0,1])
        errs.append(pvn[0,2])
        feat_vec.append([P, Q, ko, n_oo, n_oe])

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

# 4 x 9 meth
e, err = get_min_err(np.load('sam_pattern_04_09.npz'))
feat_vec.append([4, 9, 0, 0, 0])
energies.append(e)
errs.append(err)

# 4 x 9 hydroxyl
e, err = get_max_err(np.load('sam_pattern_04_09.npz'))
feat_vec.append([4, 9, 83, 50])
energies.append(e)
errs.append(err)

# 4 x 4 meth
e, err = get_min_err(np.load('sam_pattern_04_04.npz'))
feat_vec.append([4, 4, 0, 0])
energies.append(e)
errs.append(err)

# 4 x 4 hydroxyl
e, err = get_max_err(np.load('sam_pattern_04_04.npz'))
feat_vec.append([4, 4, 33, 30])
energies.append(e)
errs.append(err)


energies = np.array(energies)
feat_vec = np.array(feat_vec)
errs = np.array(errs)

np.savez_compressed('sam_pattern_pure', energies=energies, feat_vec=feat_vec, err_energies=errs)


