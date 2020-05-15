
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
    n_edge = states[0].n_edges
    # ko_int, ko_ext, n_oo [total number of oo internal edges], 
    n_feat = 3
    feat_vec = np.zeros((n_sample, n_feat))

    for i,state in enumerate(states):
        hydroxyl_mask = ~state.methyl_mask

        feat_vec[i,0] = hydroxyl_mask[state.int_indices].sum()
        #feat_vec[i,0] = hydroxyl_mask.sum()
        feat_vec[i,1] = np.dot(hydroxyl_mask, state.ext_count)

        #assert feat_vec[i].sum() == state.k_o
        feat_vec[i,2] = state.n_oo

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

def aic(n_samples, mse, k_params, do_corr=False):

    if do_corr:
        corr = (2*k_params**2 + 2*k_params) / (n_samples-k_params-1)
    else:
        corr = 0

    return n_samples*np.log(mse) + 2*k_params + corr

# Partition data into k groups
#def partition_feat(feat_vec, k=3)

### Merge edge types

energies, ols_feat_vec, states = extract_from_ds('data/sam_pattern_02_02.npz')
feat_vec1 = get_feat_vec(states)
feat_vec2 = test_get_feat_vec(states)
feat_vec3 = test_get_feat_vec2(states)


#state = states[400]
state = states[4]

# Full feature vector (edge type for each edge, 3*M_tot)
perf1, err1, _, _, reg1 = fit_k_fold(feat_vec1, energies, k=4, do_ridge=True)
# Constraints removed (indicator function for each patch node, plus internal edge types; M_int+N_tot)
perf2, err2, _, _, reg2 = fit_k_fold(feat_vec2, energies, k=4)
perf3, err3, _, _, reg3 = fit_k_fold(feat_vec3, energies, k=4)
perf_m3, err_m3, _, _, reg_m3 = fit_k_fold(ols_feat_vec, energies, k=4) 

#coef = reg.coef_.reshape((state.n_edges, 3))
coef = reg2.coef_.reshape((-1,1))

clust = AgglomerativeClustering(n_clusters=3, linkage='ward', affinity='euclidean')
clust.fit(coef)


cmap = mpl.cm.tab10
norm = plt.Normalize(0,9)
colors = cmap(norm(clust.labels_))


'''
edge_labels = np.zeros(3*state.n_edges)
edge_labels[0::3] = clust.labels_
edge_labels[1::3] = clust.labels_ + state.n_edges
edge_labels[2::3] = clust.labels_ + 2*state.n_edges

#red_feat = construct_red_feat(feat_vec, edge_labels)
plt.close()
state.plot()
state.plot_edges(do_annotate=False, colors=colors)


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
'''
