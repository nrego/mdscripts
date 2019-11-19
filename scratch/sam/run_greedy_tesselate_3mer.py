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


idx = np.arange(36)[methyl_pos[855]] 
state = State(idx)
tess = Tessalator()
can_tess = tess.is_tessalable_trimer(idx, tile_list, nn)

linewidth = np.ones(36)
linewidth[[0,1]] = 4

state.plot(linewidth=linewidth)

pos_tessalable = np.zeros(methyl_pos.shape[0], dtype=bool)

for i, methyl_mask in enumerate(methyl_pos):
    print(i)
    tess.reset()
    idx = np.arange(36)[methyl_mask]
    pos_tessalable[i] = tess.is_tessalable_trimer(idx, tile_list, nn)

# pattern: true means bead is non-polar
# tile_list[i] => j,k gives tile (i,j,k)
def make_traj(state, tile_list, pattern=[True, True, True]):
    np.random.seed()

    # Base case - we're at fully tiled state
    if state.avail_indices.size == 0:
        return

    print("\n#######")
    print("Doing N={}".format(state.avail_indices.size))
    print("########\n")
    new_states = np.array([], dtype=object)
    new_energies = np.array([])
    new_avail_indices = []


    for i in state.avail_indices:
        for j,k in tile_list[i]:
            if j not in state.avail_indices or k not in state.avail_indices:
                continue
            assert j != k
            assert i != j
            assert i != k
            # List of methyl indices for this state
            this_idx = state.pt_idx.copy()

            i_idx = np.where(state.avail_indices==i)[0].item()
            j_idx = np.where(state.avail_indices==j)[0].item()
            k_idx = np.where(state.avail_indices==k)[0].item()

            # Check that removing (i,j) results in a tile-able state
            test_new_idx = np.delete(state.avail_indices, (i_idx, j_idx, k_idx))
            #tess.reset()
            #print("  trying tile...{}".format(test_new_idx))
            can_tess = tess.is_tessalable_trimer(test_new_idx, tile_list, nn)
            if not can_tess:
                continue
            #print("    ..ok")
            if state.mode == 'build_phob':
                # Add tile (i,j,k), make appropriate spots non-polar according to our patterning scheme
                new_idx = this_idx
                if pattern[0]:
                    new_idx = np.append(new_idx, i).astype(int)
                if pattern[1]:
                    new_idx = np.append(new_idx, j).astype(int)
                if pattern[2]:
                    new_idx = np.append(new_idx, k).astype(int)

            else:
                # Add tile (i,j,k), convert appropriate spots to polar
                new_idx = this_idx
                if not pattern[0]:
                    new_idx = np.delete(new_idx, np.where(new_idx==i)[0].item())
                if not pattern[1]:
                    new_idx = np.delete(new_idx, np.where(new_idx==j)[0].item())
                if not pattern[2]:
                    new_idx = np.delete(new_idx, np.where(new_idx==k)[0].item())

            new_state = State(new_idx, reg=reg, e_func=get_energy, mode=state.mode)
            new_state._avail_indices = test_new_idx

            new_states = np.append(new_states, new_state)
            new_energies = np.append(new_energies, new_state.energy)
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
    make_traj(new_state, tile_list, pattern)

# Fully hydrophilic state; build phobicity
state_0 = State([], reg=reg, e_func=get_energy)
# Fully hydrophobic state; break phobicity
state_1 = State(np.arange(36), reg=reg, e_func=get_energy, mode='build_phil')

make_traj(state_0, tile_list, pattern=[True,True,False])
make_traj(state_1, tile_list, pattern=[True,True,False])

import pickle

with open('trimer_build_phob_bbl.dat', 'wb') as fout:
    pickle.dump(state_0, fout)

with open('trimer_build_phil_bbl.dat', 'wb') as fout:
    pickle.dump(state_1, fout)

# Fully hydrophilic state; build phobicity
state_0 = State([], reg=reg, e_func=get_energy)
# Fully hydrophobic state; break phobicity
state_1 = State(np.arange(36), reg=reg, e_func=get_energy, mode='build_phil')

make_traj(state_0, tile_list, pattern=[True,False,True])
make_traj(state_1, tile_list, pattern=[True,False,True])


with open('trimer_build_phob_blb.dat', 'wb') as fout:
    pickle.dump(state_0, fout)

with open('trimer_build_phil_blb.dat', 'wb') as fout:
    pickle.dump(state_1, fout)
#make_traj_mov(state_1)
