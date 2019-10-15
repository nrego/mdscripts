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


    def is_connected(indices, tile_list, vertices_encountered=None, start_vertex=None):
        """ determines if the graph is connected """
        if vertices_encountered is None:
            vertices_encountered = list()
        gdict = tile_list  
        vertices = indices # "list" necessary in Python 3 
        if not start_vertex:
            # chosse a vertex from graph as a starting point
            start_vertex = vertices[0]
        vertices_encountered.append(start_vertex)
        for vertex in gdict[start_vertex]:
            if vertex not in vertices_encountered:
                vertices_encountered.append(vertex)

        grp = np.unique(vertices_encountered)
        if len(vertices_encountered) == len(vertices):
            return True
        
        else:
            grp1 = np.array(vertices_encountered)
            grp2 = np.setdiff1d(vertices, vertices_encountered)
            return (grp1, grp2)

    # Find if a pattern (given by a list of indices that are still available)
    #   is tile-able
    # Not tile-able if any position does not have a partner
    def is_tessalable(self, indices, tile_list):

        # Neighbors in this pattern only
        this_tile_list = dict()
        n_neigh = []
        for i in indices:
            neigh = []
            for j in tile_list[i]:
                if j in indices:
                    neigh.append(j)
            n_neigh.append(len(neigh))
            this_tile_list[i] = neigh

        # Indices sorted by how many neighbors (how many tiles they can make)
        ordered_indices = indices[np.argsort(n_neigh)]
        #self.states.append(State(indices))
        ## Check if we can exclude this pattern right off the bat
        
        # Patterns with odd number of pieces trivially non-tileable
        if indices.size % 2 != 0:
            return False
        # Anything with an isolated piece is non-tileable
        for i in indices:
            neigh = this_tile_list[i]
            if len(neigh) == 0:
                return False

        for bad_indices in self.non_tileable:
            if np.array_equal(indices, bad_indices):
                return False

        # Base case - two indices remaining; return true if i,j form tile, else False
        if indices.size == 2:
            i, j = indices
            return (j in this_tile_list[i] or i in this_tile_list[j])

        # Now for meat:
        # Tile-able if we can decompose this pattern into a tile and another tileable pattern

        # First check for any postion i that has a single neighbor j
        #   Then, (i,j) must be a tile, so we can remove it and ask if the rest is tileable
        for i in indices:
            cnt = 0
            for j in this_tile_list[i]:
                cnt += 1

            if cnt == 1:
                i_idx = np.where(indices==i)[0].item()
                j_idx = np.where(indices==j)[0].item()

                new_indices = np.delete(indices, (i_idx, j_idx))
                new_indices.sort()
                self.states.append(State(new_indices))
                return self.is_tessalable(new_indices, this_tile_list)
                
        #return
        # No positions with single neighbor - so we have to do it the hard way
        for i in ordered_indices:
            for j in this_tile_list[i]:
                if j < i:
                    continue

                i_idx = np.where(indices==i)[0].item()
                j_idx = np.where(indices==j)[0].item()

                new_indices = np.delete(indices, (i_idx, j_idx))
                new_indices.sort()

                tessalable = self.is_tessalable(new_indices, this_tile_list)

                if tessalable:
                    self.states.append(State(new_indices))
                    return True
                else:
                    self.non_tileable.append(new_indices)

        return False

                