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
energies, methyl_pos, k_vals, positions, pos_ext, patch_indices, nn, nn_ext, edges, ext_indices, int_indices = extract_data()
# global indices of non-patch nodes
non_patch_indices = np.setdiff1d(np.arange(pos_ext.shape[0]), patch_indices)

# Locally indexed array: ext_count[i] gives number of non-patch neighbors to patch atom i
ext_count = np.zeros(36, dtype=int)
for i in range(36):
    ext_count[i] = np.intersect1d(non_patch_indices, nn_ext[i]).size
norm = plt.Normalize(-1,1)



def make_traj_mov(state):
    i = 0
    while True:
        plt.close('all')
        state.plot()
        plt.savefig('{}/Desktop/fig_{:02d}.png'.format(homedir, i))
        if len(state.children) == 0:
            break
        state = state.children[0]
        i += 1


k_eff_all_shape = np.load('k_eff_all.dat.npy')
assert len(edges) == k_eff_all_shape.shape[1]
n_conn_type = k_eff_all_shape.shape[2]

k_eff_int_edge = k_eff_all_shape[:, int_indices, :].sum(axis=1)
k_eff_ext_edge = k_eff_all_shape[:, ext_indices, :].sum(axis=1)

# n_mm, n_mo_int, n_mo_ext
feat_vec = np.dstack((k_eff_int_edge[:,0], k_eff_int_edge[:,2], k_eff_ext_edge[:,2])).squeeze(axis=0)
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)


# Fully hydrophilic state
state_0 = State([], None, reg, get_energy)
# Fully hydrophobic state
state_1 = State(np.arange(36), None, reg, get_energy, mode='build_phil')


# For each index i, give list of indices {j>i} for which ij is a tile
tile_list = dict()

for i, v in nn.items():
    new_v = [j for j in v]
    tile_list[i] = new_v


idx = np.arange(36)[methyl_pos[267]] 
state = State(idx)
tess = Tessalator()
can_tess = tess.is_tessalable(idx, tile_list)

'''

pos_tessalable = np.zeros(methyl_pos.shape[0], dtype=bool)

for i, methyl_mask in enumerate(methyl_pos):
    print(i)
    tess.reset()
    idx = np.arange(36)[methyl_mask]
    pos_tessalable[i] = tess.is_tessalable(idx, tile_list)



def make_traj(state):
    np.random.seed()
    #print(state.avail_indices.size)
    # Base case - we're at fully hydrophobic state
    if state.avail_indices.size == 0:
        return

    new_states = np.array([], dtype=object)
    new_energies = np.array([])
    for new_state in state.gen_next_pattern():
        new_states = np.append(new_states, new_state)
        new_energies = np.append(new_energies, new_state.energy)

    if state.mode == 'build_phob':
        lim_e = new_energies.min()
    else:
        lim_e = new_energies.max()

    cand_idx = new_energies == lim_e
    cand_states = new_states[cand_idx]
    print("{} candidate states going from {}".format(cand_idx.sum(), state.avail_indices.size))
    
    new_state = np.random.choice(cand_states)
    state.add_child(new_state)
    make_traj(new_state)

make_traj(state_1)
make_traj_mov(state_1)
'''