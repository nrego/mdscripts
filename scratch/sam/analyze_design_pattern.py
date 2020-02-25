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

from scipy.spatial import cKDTree

from scratch.sam.util import *

from scratch.neural_net.lib import *

import pickle

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'lines.linewidth':2})
mpl.rcParams.update({'lines.markersize':8})
mpl.rcParams.update({'legend.fontsize':10})

homedir = os.environ['HOME']

reg = np.load('sam_reg_total.npy').item()
f_c = np.dot(reg.coef_[:3], np.array([36, 6, 6]))
alpha = reg.coef_[3:]


def get_energies(edge_types, f_c, alpha):
    indices = np.array([0,1])
    energies = np.zeros((edge_types.shape[0], edge_types.shape[1]))

    for i in range(edge_types.shape[0]):
        this_edges = np.hstack((np.arange(37).reshape(-1,1), edge_types[i][:,np.array([0,1])]))
        energies[i] = np.dot(this_edges, alpha) + f_c

    return energies

# Loop thru a given (starting) state's children,
#   making not of the position of added tile at each round
#   Can optionally break out when break_out==k_c
#
#   Returns:
#       tiles: list of child tiles for each round
#       ms: Root mean sq of each tile's placement (?? I don't think this is doing what you want it to do)
#       state: Final child state
#
def get_tiles(state, break_out=None):

    indices_all = np.arange(state.N)
    prev_avail_indices = state.avail_indices

    tiles = []
    states = []
    states.append(state)

    ## Loop thru each state's children
    while len(state.children) > 0:

        state = state.children[0]

        # Find out which tile was added

        # Indices where no tiles have been placed (yet)
        this_avail_indices = state.avail_indices
        # Indices where tiles have been placed
        tile_indices = np.setdiff1d(indices_all, this_avail_indices)
        # Indices of newest tile that has just been placed on this pattern
        new_tile_indices = np.setdiff1d(prev_avail_indices, this_avail_indices)
        prev_avail_indices = this_avail_indices

        tiles.append(tuple(new_tile_indices))
        # Position(s) of new tile's headgroup(s)
        this_pos = state.positions[new_tile_indices]
        this_cent = this_pos.mean(axis=0)
        this_diff_sq = ((this_pos - this_cent)**2).sum(axis=1)
        states.append(state)

        if state.pt_idx.size == break_out:
            return tiles, np.array(ms), state


    return tiles, states, state

# Extract (n_oo, n_oe, n_cc, n_ce, n_oc) for each state
#
# Input: states (iterable of states)
def extract_feat(states):
    feat = np.zeros((len(states), 5))

    for i, state in enumerate(states):
        feat[i] = state.n_oo, state.n_oe, state.n_mm, state.n_me, state.n_mo

    return feat

def find_distance_occ(dist, bins):
    assign = np.digitize(dist, bins=bins) - 1
    occ = np.zeros(bins.size-1)
    for i in range(bins.size-1):
        occ[i] = (assign == i).sum()

    assert occ[0] == 1
    # Don't want to count distance to itself
    occ[0] = 0

    return occ

def get_rdf(state, bins, interior_only=False):
    
    tree = cKDTree(state.positions)
    # NxN sym matrix, d_mat[i,j] = d_mat[j,i] is distance between headgroup i and j
    d_mat = np.round(tree.sparse_distance_matrix(tree, max_distance=10).toarray(), 4)

    indices = np.arange(state.N)

    rdf_oo = np.zeros(bins.size-1)
    rdf_cc = np.zeros_like(rdf_oo)
    rdf_oc = np.zeros_like(rdf_oo)

    # For each head group
    for i in indices:
        # What's this for again??
        if interior_only and np.sum(d_mat[i] == 5) < 6:
            continue

        # Total number of points at given distance from headgroup i
        #   (For normalizing)
        occ = find_distance_occ(d_mat[i], bins)
        occ_mask = occ > 0

        this_rdf_oo = np.zeros_like(rdf_oo)
        this_rdf_cc = np.zeros_like(rdf_oo)
        this_rdf_oc = np.zeros_like(rdf_oc)

        i_is_methyl = (i in state.pt_idx)

        # For each other headgroup that is not i
        for j in indices:
            if j == i:
                continue

            j_is_methyl = (j in state.pt_idx)

            bin_idx = np.digitize(d_mat[i,j], bins=bins) - 1

            # o-o
            if (not i_is_methyl and not j_is_methyl):
                this_rdf_oo[bin_idx] += 1

            # c-c
            elif (i_is_methyl and j_is_methyl):
                this_rdf_cc[bin_idx] += 1

            # o-c or c-o
            else:
                this_rdf_oc[bin_idx] += 1

        # Sanity
        assert np.array_equal(occ, this_rdf_oo+this_rdf_cc+this_rdf_oc)

        rdf_oo[occ_mask] += this_rdf_oo[occ_mask] / occ[occ_mask]
        rdf_cc[occ_mask] += this_rdf_cc[occ_mask] / occ[occ_mask]
        rdf_oc[occ_mask] += this_rdf_oc[occ_mask] / occ[occ_mask]

    rdf_oo /= state.N
    rdf_cc /= state.N
    rdf_oc /= state.N

    return rdf_cc, rdf_oo, rdf_oc



### PHIL == BREAKING phobicity (k_o centric model)
fnames = glob.glob('trial_*/mono_build_phil.dat')

with open(fnames[0], 'rb') as fin:
    tmp_state = pickle.load(fin)
N = tmp_state.N
half_idx = int(3*N/4)

bins = np.arange(0, 3.8, 0.2)
rdfs_phil_oo = np.zeros((len(fnames), bins.size-1))
rdfs_phob_oo = np.zeros_like(rdfs_phil_oo)
rdfs_phil_cc = np.zeros_like(rdfs_phil_oo)
rdfs_phob_cc = np.zeros_like(rdfs_phil_oo)
rdfs_phil_oc = np.zeros_like(rdfs_phil_oo)
rdfs_phob_oc = np.zeros_like(rdfs_phil_oo)

# n_oo, n_oe, n_cc, n_ce, n_oc
edgetype_phil = np.zeros((len(fnames), N+1, 5))
edgetype_phob = np.zeros_like(edgetype_phil)

for i, fname in enumerate(fnames):
    this_dir = os.path.dirname(fname)
    print('Doing {}'.format(this_dir))

    # phob->phil (ko centric model)
    fname_phil = '{}/mono_build_phil.dat'.format(this_dir)
    # phil->phob (kc centric model)
    fname_phob = '{}/mono_build_phob.dat'.format(this_dir)

    try:
        with open(fname_phil, 'rb') as fin:
            state0 = pickle.load(fin)
        with open(fname_phob, 'rb') as fin:
            state1 = pickle.load(fin)

    except:
        print("  pickling error...")
        continue

    assert state0.N == state1.N == N

    # BREAKING phobicity
    tile0, states0, final0 = get_tiles(state0)
    # BUILDING phobicity
    tile1, states1, final1 = get_tiles(state1)

    edgetype_phil[i] = extract_feat(states0)
    edgetype_phob[i] = extract_feat(states1)

    if i % 100 == 0:
        plt.close('all')
        states0[half_idx].plot()
        plt.savefig('{}/Desktop/snap_{:03d}_phil'.format(homedir, i), transparent=True)
        plt.close('all')
        states1[half_idx].plot()
        plt.savefig('{}/Desktop/snap_{:03d}_phob'.format(homedir, i), transparent=True)
        plt.close('all')

    rdf0_cc, rdf0_oo, rdf0_oc = get_rdf(states0[half_idx], bins)
    rdf1_cc, rdf1_oo, rdf1_oc = get_rdf(states1[half_idx], bins)

    rdfs_phil_oo[i] = rdf0_oo
    rdfs_phob_oo[i] = rdf1_oo

    rdfs_phil_cc[i] = rdf0_cc
    rdfs_phob_cc[i] = rdf1_cc

    rdfs_phil_oc[i] = rdf0_oc
    rdfs_phob_oc[i] = rdf1_oc


energies_phil = get_energies(edgetype_phil, f_c, alpha)
energies_phob = get_energies(edgetype_phob[:,::-1,:], f_c, alpha)

avg_energies_phil = energies_phil.mean(axis=0)
avg_energies_phob = energies_phob.mean(axis=0)

err_energies_phil = energies_phil.std(axis=0, ddof=1)
err_energies_phob = energies_phob.std(axis=0, ddof=1)

# Average number of edge types, for each round of tesselation
#   Shape: (n_tile_rounds, n_edge_type)
# edgetypes: (n_oo, n_oe, n_cc, n_ce, n_oc)
avg_edgetype_phil = edgetype_phil.mean(axis=0)
avg_edgetype_phob = edgetype_phob.mean(axis=0)

err_edgetype_phil = edgetype_phil.std(axis=0, ddof=1)
err_edgetype_phob = edgetype_phob.std(axis=0, ddof=1)


## Plot average energies
plt.close('all')
fig, ax = plt.subplots(figsize=(6.5,6))

rounds = np.arange(N+1)
ax.errorbar(rounds, avg_energies_phil, yerr=err_energies_phil, fmt='-o', color='#1f77b4', label='breaking phobicity')
ax.errorbar(rounds, avg_energies_phob, yerr=err_energies_phob, fmt='-o', color='#7f7f7f', label='building phobicity')
ax.set_xticks(rounds[::6])
plt.savefig('{}/Desktop/avg_f'.format(homedir), transparent=True)


## Plot noo
plt.close('all')
fig, ax = plt.subplots(figsize=(6.5,6))

rounds = np.arange(N+1)
ax.errorbar(rounds, avg_edgetype_phil[:,0], yerr=err_edgetype_phil[:,0], fmt='-o', color='#1f77b4', label='breaking phobicity')
ax.errorbar(rounds, avg_edgetype_phob[::-1,0], yerr=err_edgetype_phob[:,0], fmt='-o', color='#7f7f7f', label='building phobicity')
ax.set_xticks(rounds[::6])
plt.savefig('{}/Desktop/avg_noo'.format(homedir), transparent=True)

## PLOT NCC ##
plt.close('all')
fig, ax = plt.subplots(figsize=(6.5,6))

rounds = np.arange(N+1)
ax.errorbar(rounds, avg_edgetype_phil[:,2], yerr=err_edgetype_phil[:,2], fmt='-o', color='#1f77b4', label='breaking phobicity')
ax.errorbar(rounds, avg_edgetype_phob[::-1,2], yerr=err_edgetype_phob[:,2], fmt='-o', color='#7f7f7f', label='building phobicity')
ax.set_xticks(rounds[::6])
plt.savefig('{}/Desktop/avg_ncc'.format(homedir), transparent=True)

## PLOT NOC ##
plt.close('all')
fig, ax = plt.subplots(figsize=(6.5,6))

rounds = np.arange(N+1)
ax.errorbar(rounds, avg_edgetype_phil[:,4], yerr=err_edgetype_phil[:,4], fmt='-o', color='#1f77b4', label='breaking phobicity')
ax.errorbar(rounds, avg_edgetype_phob[::-1,4], yerr=err_edgetype_phob[:,4], fmt='-o', color='#7f7f7f', label='building phobicity')
ax.set_xticks(rounds[::6])
plt.savefig('{}/Desktop/avg_noc'.format(homedir), transparent=True)

## PLOT NOE ##
plt.close('all')
fig, ax = plt.subplots(figsize=(6.5,6))

rounds = np.arange(N+1)
ax.errorbar(rounds, avg_edgetype_phil[:,1], yerr=err_edgetype_phil[:,1], fmt='-o', color='#1f77b4', label='breaking phobicity')
ax.errorbar(rounds, avg_edgetype_phob[::-1,1], yerr=err_edgetype_phob[:,1], fmt='-o', color='#7f7f7f', label='building phobicity')
ax.set_xticks(rounds[::6])
plt.savefig('{}/Desktop/avg_noe'.format(homedir), transparent=True)

#############################################
#############################################

# Mask any values that are always zero
mask_phil_oo = rdfs_phil_oo.sum(axis=0) != 0
mask_phob_oo = rdfs_phob_oo.sum(axis=0) != 0

mask_phil_cc = rdfs_phil_cc.sum(axis=0) != 0
mask_phob_cc = rdfs_phob_cc.sum(axis=0) != 0

mask_phil_oc = rdfs_phil_oc.sum(axis=0) != 0
mask_phob_oc = rdfs_phob_oc.sum(axis=0) != 0


# Plot rdfs, along with variances
# OO
plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))

ax.errorbar(bins[:-1][mask_phil_oo], rdfs_phil_oo.mean(axis=0)[mask_phil_oo], yerr=rdfs_phil_oo.std(axis=0, ddof=1)[mask_phil_oo], fmt='-o', color='#1f77b4', label='breaking phobicity')
ax.errorbar(bins[:-1][mask_phob_oo], rdfs_phob_oo.mean(axis=0)[mask_phob_oo], yerr=rdfs_phob_oo.std(axis=0, ddof=1)[mask_phob_oo], fmt='-o', color='#7f7f7f', label='building phobicity')

plt.savefig('{}/Desktop/rdf_oo.png'.format(homedir), transparent=True)

## CC
plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))

ax.errorbar(bins[:-1][mask_phil_cc], rdfs_phil_cc.mean(axis=0)[mask_phil_cc], yerr=rdfs_phil_cc.std(axis=0, ddof=1)[mask_phil_cc], fmt='-o', color='#1f77b4', label='breaking phobicity')
ax.errorbar(bins[:-1][mask_phob_cc], rdfs_phob_cc.mean(axis=0)[mask_phob_cc], yerr=rdfs_phob_cc.std(axis=0, ddof=1)[mask_phob_cc], fmt='-o', color='#7f7f7f', label='building phobicity')

plt.savefig('{}/Desktop/rdf_cc.png'.format(homedir), transparent=True)

## OC
plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))

ax.errorbar(bins[:-1][mask_phil_oc], rdfs_phil_oc.mean(axis=0)[mask_phil_oc], yerr=rdfs_phil_oc.std(axis=0, ddof=1)[mask_phil_oc], fmt='-o', color='#1f77b4', label='breaking phobicity')
ax.errorbar(bins[:-1][mask_phob_oc], rdfs_phob_oc.mean(axis=0)[mask_phob_oc], yerr=rdfs_phob_oc.std(axis=0, ddof=1)[mask_phob_oc], fmt='-o', color='#7f7f7f', label='building phobicity')

plt.savefig('{}/Desktop/rdf_oc.png'.format(homedir), transparent=True)  





