from __future__ import division, print_function


import numpy as np
import os, glob
import networkx as nx

import argparse
from IPython import embed

import cPickle as pickle
from matplotlib import pyplot as plt

from scipy.spatial import cKDTree

def gen_graph(positions, indices):

    local_indices = np.arange(indices.size)
    these_positions = positions[indices]

    tree = cKDTree(these_positions)
    local_edges = list(tree.query_pairs(r=0.6))

    graph = nx.Graph()
    graph.add_nodes_from(indices)

    for local_i, local_j in local_edges:
        global_i = indices[local_i]
        global_j = indices[local_j]
        graph.add_edge(global_i, global_j)

    return graph


positions = np.loadtxt('../../../positions.dat')

indices_ch3 = np.loadtxt('this_pt.dat', dtype=int)
indices_oh = np.setdiff1d(np.arange(36), indices_ch3)

graph_ch3 = gen_graph(positions, indices_ch3)
graph_oh = gen_graph(positions, indices_oh)

n_clust_ch3 = len(list(nx.connected_components(graph_ch3)))
n_clust_oh = len(list(nx.connected_components(graph_oh)))
embed()
