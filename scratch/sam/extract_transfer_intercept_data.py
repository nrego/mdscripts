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

### PLOT Transferability of coefs for 4x4, 6x6, 4x9 ####
#########################################

reg = np.load('reg_pooled.dat.npy')

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
        if int(pathname.split('/')[1].split('_')[1]) != P*Q:
            continue
        pvn = np.loadtxt(pathname)

        energies.append(pvn[0,1])
        errs.append(pvn[0,2])
        feat_vec.append([P*Q, P, Q])


e, err = get_min_err(np.load('sam_pattern_06_06.npz'))
feat_vec.append([36, 6, 6])
energies.append(e)
errs.append(err)

e, err = get_min_err(np.load('sam_pattern_04_09.npz'))
feat_vec.append([36, 4, 9])
energies.append(e)
errs.append(err)

e, err = get_min_err(np.load('sam_pattern_04_04.npz'))
feat_vec.append([16, 4, 4])
energies.append(e)
errs.append(err)

energies = np.array(energies)
feat_vec = np.array(feat_vec)
errs = np.array(errs)

np.savez_compressed('sam_pattern_methyl', energies=energies, feat_vec=feat_vec, err_energies=errs)


