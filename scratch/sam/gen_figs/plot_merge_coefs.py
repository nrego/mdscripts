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
mpl.rcParams.update({'legend.fontsize':10})


def get_edge_from_state(state, edges):

    edge_feat_vec = np.zeros(edges.shape[0])

    for idx, (i,j) in enumerate(edges):

        assert i in state.patch_indices

        local_i = np.where(state.patch_indices == i)[0].item()

        # Pos i is a hydroxyl
        if not state.methyl_mask[local_i]:
            # If j is not in patch, we're done
            if j not in state.patch_indices:
                edge_feat_vec[idx] = 1
            else:
                local_j = np.where(state.patch_indices == i)[0].item()
                if not state.methyl_mask[local_j]:
                    edge_feat_vec[idx] = 1

    return edge_feat_vec



ds = np.load('sam_pattern_06_06.npz')
energies = ds['energies']
states = ds['states']

temp_state = states[-1]

edges, ext_indices = enumerate_edges(state.positions, state.pos_ext, state.nn_ext, state.patch_indices)

# k_o, n_oo, n_oe
# Gen feat vec 
myfeat = np.zeros((energies.size, 3))

# k_o, n_edges 
for i, state in enumerate(states):
    myfeat[i] = state.k_o, state.n_oo, state.n_oe


