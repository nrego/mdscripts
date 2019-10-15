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


'''


# Fully hydrophilic state
state_0 = State([], None, reg, get_energy)
# Fully hydrophobic state
state_1 = State(np.arange(36), None, reg, get_energy, mode='build_phil')

def make_traj(state, tile_list):
    np.random.seed()
    #print(state.avail_indices.size)
    # Base case - we're at fully hydrophobic state
    if state.avail_indices.size == 0:
        return

    print("\n#######")
    print("Doing N={}".format(state.avail_indices.size))
    print("########\n")
    new_states = np.array([], dtype=object)
    new_energies = np.array([])
    new_avail_indices = []

    for i in state.avail_indices:
        for j in tile_list[i]:
            if j < i or j not in state.avail_indices:
                continue

            # List of methyl indices for this state
            this_idx = state.pt_idx.copy()

            i_idx = np.where(state.avail_indices==i)[0].item()
            j_idx = np.where(state.avail_indices==j)[0].item()

            # Check that removing (i,j) results in a tile-able state
            test_new_idx = np.delete(state.avail_indices, (i_idx, j_idx))
            #tess.reset()
            print("  trying tile...{}".format(test_new_idx))
            can_tess = tess.is_tessalable(test_new_idx, tile_list)
            if not can_tess:
                continue
            print("    ..ok")
            if state.mode == 'build_phob':
                # Add hydrophobic tile (i,j) where i is phobic
                new_idx_i = np.append(this_idx, i).astype(int)
                # Add tile (i,j) where j is phobic
                new_idx_j = np.append(this_idx, j).astype(int)

            else:
                # Add tile (i,j) where i is hydroxyl
                new_idx_i = np.delete(this_idx, i_idx).astype(int)
                # Add tile (i,j) where j is hydroxyl
                new_idx_j = np.delete(this_idx, j_idx).astype(int)

            new_state_i = State(new_idx_i, None, reg, get_energy, mode=state.mode)
            new_state_i._avail_indices = test_new_idx
            new_state_j = State(new_idx_j, None, reg, get_energy, mode=state.mode)
            new_state_j._avail_indices = test_new_idx
            new_states = np.append(new_states, new_state_i)
            new_states = np.append(new_states, new_state_j)
            new_energies = np.append(new_energies, new_state_i.energy)
            new_energies = np.append(new_energies, new_state_j.energy)

            new_avail_indices.append(test_new_idx)
            new_avail_indices.append(test_new_idx)

    if state.mode == 'build_phob':
        lim_e = new_energies.min()
    else:
        lim_e = new_energies.max()

    cand_idx = new_energies == lim_e
    cand_states = new_states[cand_idx]
    print("\n#################")
    print("{} candidate states going from {}".format(cand_idx.sum(), state.avail_indices.size))
    print("##################\n")
    
    new_state = np.random.choice(cand_states)
    state.add_child(new_state)
    make_traj(new_state, tile_list)

make_traj(state_0, tile_list)
make_traj_mov(state_0)
