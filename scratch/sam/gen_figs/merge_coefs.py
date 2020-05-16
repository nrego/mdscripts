
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


def unpackbits(x, num_bits):
  xshape = list(x.shape)
  x = x.reshape([-1, 1])
  to_and = 2**np.arange(num_bits).reshape([1, num_bits])

  return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

# Construct all possible states with given size
def build_states(p=2, q=2):
    n = p*q
    n_states = 2**n

    #if n > 8:
    #    print("too many states, exiting")

    states = np.empty(n_states, dtype=object)

    indices = np.arange(n)

    methyl_masks = unpackbits(np.arange(n_states), n).astype(bool)

    for i, methyl_mask in enumerate(methyl_masks):

        states[i] = State(indices[methyl_mask], p, q)

    return states


def get_rank(feat):
    d = feat - feat.mean(axis=0)
    cov = np.dot(d.T, d)

    return np.linalg.matrix_rank(cov)

# Construct a reduced feature set by summing over features
#    with like labels
def construct_red_feat(feat_vec, labels):

    n_sample = feat_vec.shape[0]
    # Number of unique features after merging based on labels
    n_clust = np.unique(labels).size

    red_feat = np.zeros((n_sample, n_clust))

    for i_clust in range(n_clust):
        mask = (labels == i_clust)
        red_feat[:, i_clust] = feat_vec[:,mask].sum(axis=1)

    return red_feat

## Get a full-edge specified feature vector
#
def get_feat_vec(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    # ko, nkoo, and nkcc, so a total of 2*M_tot + 1 coef's
    idx = 3*states[0].n_edges
    n_feat = idx
    feat_vec = np.zeros((n_sample, n_feat))

    for i,state in enumerate(states):

        feat_vec[i, 0:idx:3] = state.edge_oo
        feat_vec[i, 1:idx:3] = state.edge_cc
        feat_vec[i, 2:idx:3] = state.edge_oc
        #feat_vec[i, -1] = state.k_o


    return feat_vec


def test_get_feat_vec(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    n_feat = states[0].M_int + states[0].N_tot

    feat_vec = np.zeros((n_sample, n_feat))

    for i,state in enumerate(states):
        hydroxyl_mask = ~state.methyl_mask

        feat_vec[i,:state.M_int] = state.edge_oo[state.edges_int_indices]
        feat_vec[i,state.M_int:] = hydroxyl_mask

    return feat_vec

def test_get_feat_vec2(states):
    n_sample = states.size
    n_feat = states[0].M_int + states[0].N_ext + 1

    idx1 = states[0].M_int
    idx2 = idx1 + states[0].N_ext

    feat_vec = np.zeros((n_sample, n_feat))

    for i, state in enumerate(states):
        hydroxyl_mask = ~state.methyl_mask

        feat_vec[i, :idx1] = state.edge_oo[state.edges_int_indices]
        feat_vec[i, idx1:idx2] = state.ext_count[state.ext_indices] * hydroxyl_mask[state.ext_indices]
        feat_vec[i, idx2:] = state.k_o

    return feat_vec



# Extract edge labels from a list of edge labels
#.  for each internal edge and for each peripheral node (since external edges only need to be specified once per peripheral node)
#.  labels: shape (M_int + N_periph); first M_int labels refer to internal edges,
#.   Next N_periph labels refer to the external edges made by each peripheral node
#
# Returns: a list of new labels, one per edge, shape: (n_edges)
def merge_and_label(labels, state):
    # One unique label per peripheral node, plus
    #   one unique label per each internal edge

    int_edge_labels = labels[:state.M_int]
    periph_node_labels = labels[state.M_int:]
    # Number of unique edge types
    n_groups = np.unique(labels).size

    n_edges = state.n_edges
    new_labels = np.ones(n_edges) * np.nan

    # Assign labels for all M_int internal edges
    new_labels[state.edges_int_indices] = int_edge_labels

    # Now go over each peripheral node, find its label, and assign
    #  this label to new_labels at each of the corresponding external edges
    for i_periph_node, edge_ext_indices in enumerate(state.nodes_to_ext_edges[state.ext_indices]):
        assert np.isnan(new_labels[edge_ext_indices]).all()
        new_labels[edge_ext_indices] = periph_node_labels[i_periph_node]

    return new_labels


def aic(n_samples, mse, k_params, do_corr=False):

    if do_corr:
        corr = (2*k_params**2 + 2*k_params) / (n_samples-k_params-1)
    else:
        corr = 0

    return n_samples*np.log(mse) + 2*k_params + corr

def plot_1d(coef, labels=None):
    if labels is None:
        labels = np.ones_like(coef)
    plt.close()
    plt.scatter(coef, np.ones_like(coef), c=labels)

# Partition data into k groups
#def partition_feat(feat_vec, k=3)

### Merge edge types
k_cv=5
energies, ols_feat_vec, states = extract_from_ds('data/sam_pattern_06_06.npz')
feat_vec1 = get_feat_vec(states)
feat_vec2 = test_get_feat_vec(states)
feat_vec3 = test_get_feat_vec2(states)


state = states[200]

# Full feature vector (edge type for each edge, 3*M_tot)
perf1, err1, _, _, reg1 = fit_k_fold(feat_vec1, energies, k=k_cv, do_ridge=True)
# Constraints removed (indicator function for each patch node, plus internal edge types; M_int+N_tot)
perf2, err2, _, _, reg2 = fit_k_fold(feat_vec2, energies, k=k_cv)
# ko, sum over all internal edges, sum over all external edges (miext edges for each peripheral node); shape: (M_int + N_periph + 1)
perf3, err3, _, _, reg3 = fit_k_fold(feat_vec3, energies, k=k_cv)
perf_m3, err_m3, _, _, reg_m3 = fit_k_fold(ols_feat_vec, energies, k=k_cv) 

## Cluster edge coefficients (leave ko coef alone)
n_clust = 3
coef = reg3.coef_[:-1].reshape((-1,1))

mask = reg3.coef_[:-1] > 0

clust = AgglomerativeClustering(n_clusters=n_clust, linkage='ward', affinity='euclidean')
clust.fit(coef)

# Presumably ko, n_oo1, n_oo2; where 1 and 2 indicate total number of oo edges of different classes,
#   determined by clustering
test_labels = np.zeros(mask.size)
test_labels[mask] = 1
red_feat = construct_red_feat(feat_vec3, np.append(clust.labels_, n_clust))
perf_red, err_red, _, _, reg_red = fit_k_fold(red_feat, energies, k=k_cv)
new_labels = merge_and_label(clust.labels_, state)

cmap = mpl.cm.Set1
norm = plt.Normalize(0,9)
colors = cmap(norm(new_labels))



