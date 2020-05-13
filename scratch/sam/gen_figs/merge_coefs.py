
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
    n_clust = np.unique(labels).size

    red_feat = np.zeros((feat_vec.shape[0], n_clust))

    new_set = []
    for i_clust in range(n_clust):
        mask = labels == i_clust
        new_set.append(feat_vec[:,mask].sum(axis=1))

    return np.vstack(new_set).T

## Get a full-edge specified feature vector
#
def get_feat_vec(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    # ko, nkoo, and nkcc, so a total of 2*M_tot + 1 coef's
    idx = 2*states[0].n_edges
    n_feat = 2*states[0].N_tot + idx
    feat_vec = np.zeros((n_sample, n_feat))

    for i,state in enumerate(states):

        feat_vec[i,:idx:2] = state.edge_oo
        feat_vec[i,1:idx:2] = state.edge_oc
        #feat_vec[i,2:idx:3] = state.edge_oc
        feat_vec[i,idx::2] = ~state.methyl_mask
        feat_vec[i,idx+1::2] = 0


    return feat_vec

def test_get_feat_vec(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    #n_feat = 2*states[0].M_int + states[0].M_ext
    n_feat = 2*states[0].M_int + states[0].N_ext

    idx1 = 2*states[0].M_int
    feat_vec = np.zeros((n_sample, n_feat))


    for i,state in enumerate(states):
        hydroxyl_mask = ~state.methyl_mask

        feat_vec[i,:idx1:2] = state.edge_oo[state.edges_int_indices]
        feat_vec[i,1:idx1:2] = state.edge_cc[state.edges_int_indices]
        #feat_vec[i, idx1:] = state.edge_oo[state.edges_ext_indices]
        feat_vec[i, idx1:] = (state.ext_count * hydroxyl_mask)[state.nodes_peripheral]
    
    return feat_vec

def test_get_feat_vec2(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    #n_feat = 2*states[0].M_int + states[0].M_ext
    n_feat = states[0].M_int + states[0].N_int + states[0].N_ext

    feat_vec = np.zeros((n_sample, n_feat))


    for i,state in enumerate(states):
        hydroxyl_mask = ~state.methyl_mask

        feat_vec[i,:state.M_int] = state.edge_oo[state.edges_int_indices]
        feat_vec[i, state.M_int:state.M_int+state.N_int] = hydroxyl_mask[state.int_indices]
        feat_vec[i, -state.N_ext:] = (state.ext_count * hydroxyl_mask)[state.nodes_peripheral]

    return feat_vec

def label_edges(labels, state):

    colors = ['k' if l==0 else 'r' for l in labels]


    return np.array(colors)


# Merge each peripheral node's edges into single group
#.  (label all external edges made by each peripheral node by its own label)
def merge_and_label(state):
    # One unique label per peripheral node, plus
    #   one unique label per each internal edge
    n_labels = state.N_ext + state.M_int
    labels = np.zeros(state.n_edges)
    labels[:] = -1

    labels[state.edges_int_indices] = np.arange(state.M_int)

    curr_label = state.M_int

    for i, ext_idx in enumerate(state.nodes_to_ext_edges):
        if ext_idx.size == 0:
            continue

        assert ext_idx.size == state.ext_count[i]
        labels[ext_idx] = curr_label
        curr_label += 1

    return labels

# Partition data into k groups
#def partition_feat(feat_vec, k=3)

### Merge edge types

energies, ols_feat_vec, states = extract_from_ds('data/sam_pattern_06_06.npz')
feat_vec = get_feat_vec(states)

#perf_mse, err, xvals, fit, reg = fit_k_fold(new_feat_vec, energies)
#perf_mse, err, xvals, fit, reg = fit_leave_one(new_feat_vec, energies)
reg = linear_model.Ridge()
#reg = linear_model.Lasso()
reg.fit(feat_vec, energies)


state = states[10]

coef = reg.coef_.reshape((state.n_edges, 3))

clust = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
clust.fit(coef)


states = build_states(3,2)
state = states[len(states)//2]
test_feat = test_get_feat_vec(states)
meta_feat = np.hstack((test_feat[:,:2*state.M_int:2], test_feat[:,2*state.M_int:]))
#meta_feat = test_feat[:,2*state.M_int:]
test_reg = linear_model.LinearRegression()
scores = []
for i in np.arange(1, 2*state.M_int, 2):
    this_y = test_feat[:, i]
    test_reg.fit(meta_feat, this_y)

    r2 = test_reg.score(meta_feat, this_y)
    scores.append(r2)

scores = np.array(scores)

cmap = mpl.cm.tab10
norm = plt.Normalize(0,9)

labels = np.zeros(state.n_edges)
labels[state.edges_ext_indices] = 1

colors = cmap(norm(labels))


