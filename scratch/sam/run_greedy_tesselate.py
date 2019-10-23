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
    indices_all = np.arange(36)
    prev_avail_indices = state.avail_indices

    plt.close('all')
    state.plot()
    plt.savefig('{}/Desktop/fig_{:02d}.png'.format(homedir, i))
    
    while True:
        if len(state.children) == 0:
            break
        state = state.children[0]
        i += 1

        # Find out which tile was added
        this_avail_indices = state.avail_indices
        tile_indices = np.setdiff1d(indices_all, this_avail_indices)
        new_tile_indices = np.setdiff1d(prev_avail_indices, this_avail_indices)
        prev_avail_indices = this_avail_indices

        linewidth = np.ones(36)
        linewidth[tile_indices] = 4

        edgecolors = np.array(['k' for i in range(36)])
        edgecolors[new_tile_indices] = 'r'

        zorders = np.ones(36)
        zorders[new_tile_indices] = 3

        plt.close('all')
        state.plot(linewidth=linewidth, edgecolors=edgecolors)
        plt.savefig('{}/Desktop/fig_{:02d}.png'.format(homedir, i))



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


idx = np.arange(36)[methyl_pos[855]] 
state = State(idx)
tess = Tessalator()
can_tess = tess.is_tessalable(idx, tile_list)

linewidth = np.ones(36)
linewidth[[0,1]] = 4

state.plot(linewidth=linewidth)

pos_tessalable = np.zeros(methyl_pos.shape[0], dtype=bool)

for i, methyl_mask in enumerate(methyl_pos):
    print(i)
    #tess.reset()
    idx = np.arange(36)[methyl_mask]
    pos_tessalable[i] = tess.is_tessalable(idx, tile_list)

tess.reset()

# Pattern is phobic-philic for each domino (i,j)
#   Note if (i,j) then (j,i) also exists, so we have both patterns
# If do_dfs, (depth-first search), then exhaustively search all greedy paths
#   (i.e. don't choose one radomly)
def make_traj(state, tile_list, do_dfs=False, prefix=None):

    np.random.seed()

    # Base case - we're at fully tiled state
    if state.avail_indices.size == 0:
        return

    if prefix is not None:
        print("\n{}\n".format(prefix))

    n_tabs = (36 - state.avail_indices.size) // 2
    pre_tab = ' '.join(['' for i in range(n_tabs)])
    print("\n{}#######".format(pre_tab))
    print("{}Doing N={}".format(pre_tab, state.avail_indices.size))
    print("{}########\n".format(pre_tab))
    new_states = np.array([], dtype=object)
    new_energies = np.array([])
    new_avail_indices = []

    # List of methyl indices for this state
    this_idx = state.pt_idx.copy()

    for i in state.avail_indices:
        for j in tile_list[i]:
            if j not in state.avail_indices:
                continue

            i_idx = np.where(state.avail_indices==i)[0].item()
            j_idx = np.where(state.avail_indices==j)[0].item()

            # Check that removing (i,j) results in a tile-able state
            test_new_idx = np.delete(state.avail_indices, (i_idx, j_idx))
            #print("  trying tile...{}".format(test_new_idx))
            can_tess = tess.is_tessalable(test_new_idx, tile_list)
            if not can_tess:
                continue
            #print("    ..ok")
            
            # Converting philic=>phobic
            if state.mode == 'build_phob':
                new_idx = np.append(this_idx, i).astype(int)

            # Converting phobic=>philic
            else:
                new_idx = np.delete(this_idx, np.where(this_idx==j)[0].item()).astype(int)

            new_state = State(new_idx, None, reg, get_energy, mode=state.mode)
            new_state._avail_indices = test_new_idx
            new_states = np.append(new_states, new_state)
            new_energies = np.append(new_energies, new_state.energy)

            new_avail_indices.append(test_new_idx)

    # find min/max energy trial move(s)
    if state.mode == 'build_phob':
        lim_e = new_energies.min()
    else:
        lim_e = new_energies.max()

    cand_idx = new_energies == lim_e
    cand_states = new_states[cand_idx]
    print("\n{}#################".format(pre_tab))
    print("{}{} candidate states going from {}".format(pre_tab, cand_idx.sum(), state.avail_indices.size))
    print("{}##################\n".format(pre_tab))
    
    if not do_dfs:
        new_state = np.random.choice(cand_states)
        #new_state = new_states[0]
        state.add_child(new_state)
        make_traj(new_state, tile_list)

    # Trace each candidate state
    else:
        n_cand = cand_idx.sum()
        for i, new_state in enumerate(cand_states):
            if state.avail_indices.size == 36:
                prefix = '{} of {} ({})'.format(i+1, n_cand, tess.iter_count)
            print("{}  Doing candidate {} of {} for N={}".format(pre_tab, i+1, n_cand, state.avail_indices.size))
            state.add_child(new_state)
            make_traj(new_state, tile_list, do_dfs, prefix)

# Fully hydrophilic state
state_0 = State([], None, reg, get_energy)
# Fully hydrophobic state
state_1 = State(np.arange(36), None, reg, get_energy, mode='build_phil')

make_traj(state_0, tile_list, do_dfs=True)


make_traj_mov(state_0)
