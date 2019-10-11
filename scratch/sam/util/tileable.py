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

class Tessalator:
    def __init__(self):
        self.states = []
        self.non_tileable = []


    def reset(self):
        self.states = []
        self.non_tileable = []

    # Find if a pattern (given by a list of indices that are still available)
    #   is tile-able
    # Not tile-able if any position does not have a partner
    def is_tessalable(self, indices, tile_list):

        #self.states.append(State(indices))
        ## Check if we can exclude this pattern right off the bat
        
        # Patterns with odd number of pieces trivially non-tileable
        if indices.size % 2 != 0:
            return False
        # Anything with an isolated piece is non-tileable
        for i in indices:
            neigh = tile_list[i]
            if len(neigh) == 0:
                continue
            has_neighbor = False
            for j in neigh:
                if j in indices:
                    has_neighbor = True
                    break
            if not has_neighbor:
                return False

        for bad_indices in self.non_tileable:
            if np.array_equal(indices, bad_indices):
                return False

        # Base case - two indices remaining; return true if i,j form tile, else False
        if indices.size == 2:
            i, j = indices
            return j in tile_list[i] or i in tile_list[j]

        # Now for meat:
        # Tile-able if we can decompose this pattern into a tile and another tileable pattern

        # First check for any postion i that has a single neighbor j
        #   Then, (i,j) must be a tile, so we can remove it and ask if the rest is tileable
        for i in indices:
            cnt = 0
            for k in tile_list[i]:
                if k in indices: 
                    cnt += 1
                    j = k

            if cnt == 1:
                i_idx = np.where(indices==i)[0].item()
                j_idx = np.where(indices==j)[0].item()

                new_indices = np.delete(indices, (i_idx, j_idx))
                new_indices.sort()
                self.states.append(State(new_indices))
                return self.is_tessalable(new_indices, tile_list)
                
        #return
        # No positions with single neighbor - so we have to do it the hard way
        for i in indices:
            for j in tile_list[i]:
                if j < i:
                    continue
                if j in indices:
                    i_idx = np.where(indices==i)[0].item()
                    j_idx = np.where(indices==j)[0].item()

                    new_indices = np.delete(indices, (i_idx, j_idx))
                    new_indices.sort()

                    # I assume this is a short-circuit and...
                    tessalable = self.is_tessalable(new_indices, tile_list)

                    if tessalable:
                        self.states.append(State(new_indices))
                        return True
                    else:
                        self.non_tileable.append(new_indices)
                