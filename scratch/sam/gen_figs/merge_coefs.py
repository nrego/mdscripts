
import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

import itertools

from sklearn import datasets, linear_model
from sklearn.cluster import AgglomerativeClustering


def get_rank(feat):
    d = feat - feat.mean(axis=0)
    cov = np.dot(d.T, d)

    return np.linalg.matrix_rank(cov)

# Full edge feat vec
#
# 0-20: ~methyl_mask [is each buried node position an o or not?]
# 36-(36+85): Identity of each internal edge [1: o-o, 0: c-c or o-c]
# (36+85):(36+85+n_node_peripheral): Number of o-e edges made by each periph node
def get_new_feat_vec(states):

    n_sample = states.size
    n_edge = states[0].n_edges
    n_edge_int = states[0].M_int
    n_feat = states[0].N_int + states[0].M_int + states[0].nodes_peripheral.sum()

    # ko, nkoo, and nkcc, so a total of 2*M_tot + 1 coef's
    feat_vec = np.zeros((n_sample, n_feat))

    for i,state in enumerate(states):

        feat_vec[i, :state.N_int] = ~(state.methyl_mask[state.nodes_buried])
        feat_vec[i, state.N_int:state.N_int+n_edge_int] = state.edge_oo[state.edges_int_indices]
        
        # Number of o-o external edges made by each periph node
        periph_oo = state.ext_count[state.nodes_peripheral] * ~state.methyl_mask[state.nodes_peripheral]
        assert periph_oo.sum() == state.n_oe

        feat_vec[i, state.N_int+n_edge_int:] = periph_oo

    return feat_vec

# Construct a reduced feature set by summing over features
#    with like labels
def construct_red_feat(feat_vec, labels):
    n_clust = np.unique(labels).size

    red_feat = np.zeros((feat_vec.shape[0], n_clust))

    new_set = []
    for i_clust in range(n_clust):
        mask = labels == i_clust
        new_set.append(feat_vec[:,mask].sum(axis=1))

    return np.vstack(new_set).T

## Get a full-edge specified feature vector
#
#  first M_int [# internal edges]: is each edge a o-o or no? [1 if oo, 0 otherwise]
#  next N_ext [# peripheral nodes]: number of o-o edges made by each peripheral node
#  last: ko
def get_full_feat(states):

    n_feat = states[0].M_int + states[0].N_ext + 1
    new_feat = np.zeros((len(states), n_feat))
    new_feat[...] = np.nan

    for i, state in enumerate(states):
        ko = state.k_o
        buried_oo = state.edge_oo[state.edges_int_indices]
        # Number of oo external edges for each peripheral node
        periph_oo = state.ext_count[state.nodes_peripheral] * ~state.methyl_mask[state.nodes_peripheral]
        
        assert state.n_oo == buried_oo.sum()
        assert state.n_oe == periph_oo.sum()

        new_feat[i, :state.M_int] = buried_oo
        new_feat[i, state.M_int:state.M_int+state.N_ext] = periph_oo
        assert np.isnan(new_feat[i,-1])
        new_feat[i,-1] = ko

    return new_feat

def label_edges(labels, state):

    edges = state.edges
    edges_int_indices = state.edges_int_indices
    edges_ext_indices = state.edges_ext_indices
    global_periph_indices = state.patch_indices[state.ext_indices]
    # First M_int are all internal edges
    n_int_edges = edges_int_indices.size
    lnames = np.unique(labels)
    n_clust = lnames.size

    mapping = dict()
    mapping[lnames[0]] = 'k'
    mapping[lnames[1]] = 'g'

    colors = np.empty(edges.shape[0], dtype=object)

    # Go over all internal edges
    for i,l in enumerate(labels[:n_int_edges]):
        # Find the index of this edge
        edge_idx = edges_int_indices[i]
        colors[edge_idx] = mapping[l]

    # Now go over peripheral nodes, find their edges
    for i,l in enumerate(labels[n_int_edges:]):
        global_i = global_periph_indices[i]

        for j, edge in enumerate(edges):
            if j in edges_ext_indices and edge[0] == global_i:
                colors[j] = mapping[l]

    return colors

def get_test_feat(states):

    n_feat = 3*states[0].n_edges
    new_feat = np.zeros((len(states), n_feat))
    new_feat[...] = np.nan

    for i, state in enumerate(states):

        new_feat[i, ::3] = state.edge_oo
        new_feat[i, 1::3] = state.edge_cc
        new_feat[i, 2::3] = state.edge_oc


    return new_feat
    
# Partition data into k groups
#def partition_feat(feat_vec, k=3)

### Merge edge types

energies, ols_feat_vec, states = extract_from_ds('data/sam_pattern_06_06.npz')
new_feat_vec = get_full_feat(states)

perf_mse, err, xvals, fit, reg = fit_k_fold(new_feat_vec, energies)

state = states[400]

coef = reg.coef_[:-1]
coef[state.M_int:] /= state.ext_count[state.nodes_peripheral]

coef = coef.reshape(-1,1)

clust = AgglomerativeClustering(n_clusters=2, linkage='ward', affinity='euclidean')
clust.fit(coef)

