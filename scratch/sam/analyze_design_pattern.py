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



def get_tiles(state, break_out=None):

    indices_all = np.arange(36)
    prev_avail_indices = state.avail_indices

    tiles = []
    ms = []
    while len(state.children) > 0:
        state = state.children[0]
        # Find out which tile was added
        this_avail_indices = state.avail_indices
        tile_indices = np.setdiff1d(indices_all, this_avail_indices)
        new_tile_indices = np.setdiff1d(prev_avail_indices, this_avail_indices)
        prev_avail_indices = this_avail_indices

        tiles.append(tuple(new_tile_indices))
        this_pos = state.positions[new_tile_indices]
        this_cent = this_pos.mean(axis=0)
        this_diff_sq = ((this_pos - this_cent)**2).sum(axis=1)
        ms.append(this_diff_sq.mean())

        if state.pt_idx.size == break_out:
            return tiles, state

    return tiles, np.array(ms), state

def find_distance_occ(dist, bins):
    assign = np.digitize(dist, bins=bins) - 1
    occ = np.zeros(bins.size-1)
    for i in range(bins.size-1):
        occ[i] = (assign == i).sum()

    return occ

def get_rdf(state, bins, interior_only=False, mode='mm'):
    
    tree = cKDTree(state.positions)
    d_mat = np.round(tree.sparse_distance_matrix(tree, max_distance=10).toarray(), 4)

    indices = np.arange(36)

    rdf_mm = np.zeros((indices.size, bins.size-1))
    rdf_oo = np.zeros((indices.size, bins.size-1))
    rdf_mo = np.zeros((indices.size, bins.size-1))

    # For each methyl group...
    for this_idx, i in enumerate(indices):
        if interior_only and np.sum(d_mat[i] == 5) < 6:
            continue
        occ = find_distance_occ(d_mat[i], bins)

        i_is_methyl = i in state.pt_idx

        # For each other methyl that is not i
        for j in indices:
            if i == j:
                continue

            bin_idx = np.digitize(d_mat[i,j], bins=bins) - 1

            # Find pair type
            # m-m
            if i_is_methyl and j in state.pt_idx:
                rdf_mm[this_idx, bin_idx] += 1

            # o-o
            elif not i_is_methyl and j not in state.pt_idx:
                rdf_oo[this_idx, bin_idx] += 1

            # m-o or o-m
            else:
                rdf_mo[this_idx, bin_idx] += 1

        rdf_mm[this_idx] = rdf_mm[this_idx] / occ
        rdf_oo[this_idx] = rdf_oo[this_idx] / occ
        rdf_mo[this_idx] = rdf_mo[this_idx] / occ

        if i_is_methyl:
            rdf_oo[this_idx,...] = np.nan
        else:
            rdf_mm[this_idx,...] = np.nan

    rdf_mm = np.ma.masked_invalid(rdf_mm).mean(axis=0)
    rdf_oo = np.ma.masked_invalid(rdf_oo).mean(axis=0)
    rdf_mo = np.ma.masked_invalid(rdf_mo).mean(axis=0)


    return rdf_mm, rdf_oo, rdf_mo




fnames = glob.glob('trial_*/trimer_build_phob_bbl.dat')

bins = np.arange(0, 3.8, 0.2)
rdfs_phob_mm = np.zeros((len(fnames), bins.size-1))
rdfs_phil_mm = np.zeros_like(rdfs_phob_mm)
rdfs_phob_oo = np.zeros_like(rdfs_phob_mm)
rdfs_phil_oo = np.zeros_like(rdfs_phob_mm)
rdfs_phob_mo = np.zeros_like(rdfs_phob_mm)
rdfs_phil_mo = np.zeros_like(rdfs_phob_mm)

## n_ext, n_curv, n_collapsed
tiles_phob = np.zeros((len(fnames), 3))
tiles_phil = np.zeros_like(tiles_phob)

for i, fname in enumerate(fnames):
    this_dir = os.path.dirname(fname)
    print('Doing {}'.format(this_dir))
    fname_phil = '{}/trimer_build_phil_bbl.dat'.format(this_dir)

    try:
        with open(fname, 'rb') as fin:
            state0 = pickle.load(fin)
        with open(fname_phil, 'rb') as fin:
            state1 = pickle.load(fin)

    except:
        print("  pickling error...")
        continue

    tile0, rms0, final0 = get_tiles(state0)
    tile1, rms1, final1 = get_tiles(state1)

    tiles_phob[i] = (rms0 > 0.16).sum(), ((rms0 < 0.16) & (rms0 > 0.1)).sum(), (rms0 < 0.1).sum()
    tiles_phil[i] = (rms1 > 0.16).sum(), ((rms1 < 0.16) & (rms1 > 0.1)).sum(), (rms1 < 0.1).sum()


    if i % 100 == 0:
        plt.close('all')
        final0.plot()
        plt.savefig('snap_{:03d}_phob'.format(i))
        plt.close('all')
        final1.plot()
        plt.savefig('snap_{:03d}_phil'.format(i))
        plt.close('all')

    rdf0_mm, rdf0_oo, rdf0_mo = get_rdf(final0, bins, mode='mo')
    rdf1_mm, rdf1_oo, rdf1_mo = get_rdf(final1, bins, mode='mo')

    rdfs_phob_mm[i] = rdf0_mm
    rdfs_phob_mm[i, rdf0_mm.mask] = np.nan
    rdfs_phil_mm[i] = rdf1_mm
    rdfs_phil_mm[i, rdf1_mm.mask] = np.nan

    rdfs_phob_oo[i] = rdf0_oo
    rdfs_phob_oo[i, rdf0_oo.mask] = np.nan
    rdfs_phil_oo[i] = rdf1_oo
    rdfs_phil_oo[i, rdf1_oo.mask] = np.nan

    rdfs_phob_mo[i] = rdf0_mo
    rdfs_phob_mo[i, rdf0_mo.mask] = np.nan
    rdfs_phil_mo[i] = rdf1_mo
    rdfs_phil_mo[i, rdf1_mo.mask] = np.nan

tot_phob_mm = np.ma.masked_invalid(rdfs_phob_mm).mean(axis=0)
tot_phil_mm = np.ma.masked_invalid(rdfs_phil_mm).mean(axis=0)

err_phob_mm = np.ma.masked_invalid(rdfs_phob_mm).std(axis=0, ddof=1)
err_phil_mm = np.ma.masked_invalid(rdfs_phil_mm).std(axis=0, ddof=1)

tot_phob_oo = np.ma.masked_invalid(rdfs_phob_oo).mean(axis=0)
tot_phil_oo = np.ma.masked_invalid(rdfs_phil_oo).mean(axis=0)

err_phob_oo = np.ma.masked_invalid(rdfs_phob_oo).std(axis=0, ddof=1)
err_phil_oo = np.ma.masked_invalid(rdfs_phil_oo).std(axis=0, ddof=1)

tot_phob_mo = np.ma.masked_invalid(rdfs_phob_mo).mean(axis=0)
tot_phil_mo = np.ma.masked_invalid(rdfs_phil_mo).mean(axis=0)

err_phob_mo = np.ma.masked_invalid(rdfs_phob_mo).std(axis=0, ddof=1)
err_phil_mo = np.ma.masked_invalid(rdfs_phil_mo).std(axis=0, ddof=1)

tot_phob = tot_phob_oo
err_phob = err_phob_oo
tot_phil = tot_phil_oo
err_phil = err_phil_oo
plt.errorbar(bins[:-1][~tot_phob.mask], tot_phob[~tot_phob.mask], yerr=err_phob[~tot_phob.mask], fmt='-o')
plt.errorbar(bins[:-1][~tot_phil.mask], tot_phil[~tot_phil.mask], yerr=err_phil[~tot_phil.mask], fmt='-o')


