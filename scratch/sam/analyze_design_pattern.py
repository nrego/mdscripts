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

with open("trimer_build_phil_bbl.dat", "rb") as fin: 
    state = pickle.load(fin)

def get_tiles(state):

    indices_all = np.arange(36)
    prev_avail_indices = state.avail_indices

    tiles = []
    while len(state.children) > 0:
        state = state.children[0]
        # Find out which tile was added
        this_avail_indices = state.avail_indices
        tile_indices = np.setdiff1d(indices_all, this_avail_indices)
        new_tile_indices = np.setdiff1d(prev_avail_indices, this_avail_indices)
        prev_avail_indices = this_avail_indices

        tiles.append(tuple(new_tile_indices))

    return tiles, state

def find_distance_occ(dist, bins):
    assign = np.digitize(dist, bins=bins) - 1
    occ = np.zeros(bins.size-1)
    for i in range(bins.size-1):
        occ[i] = (assign == i).sum()

    return occ

def get_rdf(state):
    bins = np.arange(0, 3.8, 0.2)
    tree = cKDTree(state.positions)
    d_mat = np.round(tree.sparse_distance_matrix(tree, max_distance=10).toarray(), 4)

    rdf = np.zeros((state.pt_idx.size, bins.size-1))

    for this_idx, i in enumerate(state.pt_idx):
        occ = find_distance_occ(d_mat[i], bins)
        for j in state.pt_idx:
            if i == j:
                continue

            bin_idx = np.digitize(d_mat[i,j], bins=bins) - 1


            rdf[this_idx, bin_idx] += 1 

        rdf[this_idx] = rdf[this_idx] / occ

    return bins, rdf.mean(axis=0)

