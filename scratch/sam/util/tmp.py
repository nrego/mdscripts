from __future__ import division

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

homedir = os.environ["HOME"]

norm = plt.Normalize(-1,1)

def plot_hex_from_pt_idx(pt_idx, ny=6, nz=6):
    pos_ext = gen_pos_grid(ny, nz)
    feat = np.zeros(pos_ext.shape[0])
    feat[:] = -1
    feat[pt_idx] = 1

    feat = feat.reshape(ny, nz).T[::-1, ::-1] 
    plot_hextensor(feat[None, None, ...], norm=norm)
    #plt.show()

p_q = np.array([(2,2),
                (3,3),
                (4,4),
                (5,5),
                (6,6),
                (7,7),
                (8,8),
                (9,9),
                (2,8),
                (2,12),
                (2,18),
                (2,32),
                (3,8),
                (3,12),
                (4,6),
                (4,9)])
'''
for p,q in p_q:
    n = p*q
    pt_idx_meth = np.ones(n, dtype=bool)
    pt_idx_oh = ~pt_idx_meth

    plt.close('all')
    plot_hex_from_pt_idx(pt_idx_meth, ny=p, nz=q)
    plt.savefig('/Users/nickrego/Desktop/fig_{:02d}_{:02d}_meth.pdf'.format(p,q))

    plt.close('all')
    plot_hex_from_pt_idx(pt_idx_oh, ny=p, nz=q)
    plt.savefig('/Users/nickrego/Desktop/fig_{:02d}_{:02d}_ih.pdf'.format(p,q))
'''

def get_rms(pos):
    if pos.shape[0] == 0:
        return np.nan

    return np.sqrt(pos.var(axis=0).sum())

dat = np.load("sam_pattern_data.dat.npz")

positions = dat['positions']
methyl_pos = dat['methyl_pos'][441:]
energies = dat['energies'][441:]
rmses = np.zeros_like(energies)

mask = methyl_pos.sum(axis=1) == 5

bins_d = np.arange(0, 2.05, 0.05)
bins_k = np.arange(0, 38)

xx, yy = np.meshgrid(bins_d, bins_k)
all_ener = np.zeros_like(xx)
all_ener[:] = np.nan

for i, methyl_mask in enumerate(methyl_pos):
    this_rms = get_rms(positions[methyl_mask])
    this_ener = energies[i]
    this_k = methyl_mask.sum()

    rms_idx = np.digitize(this_rms, bins_d) - 1
    k_idx = np.digitize(this_k, bins_k) - 1

    all_ener[k_idx, rms_idx] = this_ener

