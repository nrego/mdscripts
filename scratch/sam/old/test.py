from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

import time
import itertools
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from scratch.sam.util import *

from scratch.neural_net.lib import *

homedir = os.environ["HOME"]
energies, methyl_pos, k_vals, positions, pos_ext, patch_indices, nn, nn_ext, edges, ext_indices, int_indices = extract_data()
# global indices of non-patch nodes
non_patch_indices = np.setdiff1d(np.arange(pos_ext.shape[0]), patch_indices)

# Locally indexed array: ext_count[i] gives number of non-patch neighbors to patch atom i
ext_count = np.zeros(36, dtype=int)
for i in range(36):
    ext_count[i] = np.intersect1d(non_patch_indices, nn_ext[i]).size
norm = plt.Normalize(-1,1)


def dfs_3(idx, nn, vis=None):

    # Maps two integers to unique number
    cantor = lambda j,k : 0.5*(j+k)*(j+k)+k

    if vis is None:
        vis = {cantor(j,k): False for (j,k) in itertools.permutations(range(36),2)}

    for j in nn[idx]:
        if j == i:
            continue
        for k in nn[j]:
            #if k == i or k == j or vis[cantor(j,k)] or vis[cantor(k,j)]:
            #    continue
            if k == i or k == j:
                continue
            vis[cantor(j,k)] = True
            vis[cantor(k,j)] = True

            #yield sorted([j,k])
            yield (j,k)
            
# For each index i, give list of indices {j>i} for which ij is a tile
tile_list = dict()

for i, v in nn.items():
    new_v = [(j,k) for j,k in dfs_3(i, nn)]
    tile_list[i] = new_v


arr = np.array([ 0,  1,  2,  3,  4,  5,  6,  7, 10, 11, 12, 13, 14, 16, 18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])

