
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

def get_feat_vec_oo(states):
    n_sample = states.size
    n_edges = states[0].n_edges
    n_feat = n_edges + 1

    feat_vec = np.zeros((n_sample, n_feat))

    for i, state in enumerate(states):
        feat_vec[i, :n_edges] = state.edge_oo
        feat_vec[i, -1] = state.k_o

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
#.  for each internal edge and for each peripheral node 
#      (since external edges only need to be specified once per peripheral node)
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

    return new_labels.astype(int)


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

# For each edge k, find its equivalent edge (thru symmetry) l
def find_sym_edges(state):
    p, q = state.p, state.q
    # Lut of symmetric nodes.
    #    node (local) patch index i gives (local) patch index j, its equivalent node
    #    This numbering is dependent on whether p is odd or even
    rev_node_lut = np.zeros(state.N_tot, dtype=int)

    # Even p, this is the easy case - can just reverse numbering
    if p % 2 == 0:
        rev_node_lut = state.N_tot - np.arange(state.N_tot) - 1

    # odd p; vertical sym axis down middle. 
    else:
        indices = np.arange(state.N_tot)
        for i_col in range(p):
            # index of equivalent column
            rev_col_idx = p-1-i_col
            rev_node_lut[i_col*q:(i_col+1)*q] = indices[rev_col_idx*q:(rev_col_idx+1)*q]

    # Edge lookup, indexed by edge k, gives sym edge l
    rev_edge_lut = np.zeros(state.n_edges, dtype=int)

    # Memoizing each external edge we visit for each (local) i
    n_ext_indices = np.zeros(state.N_tot, dtype=int)

    for idx_k, (i,j) in enumerate(state.edges):
        local_i = np.argwhere(i==state.patch_indices)[0].item()

        # (local) index of i's symmetric node
        rev_local_i = rev_node_lut[local_i]
        rev_global_i = state.patch_indices[rev_local_i]

        j_int = j in state.patch_indices

        # internal edge
        if j_int:
            local_j = np.argwhere(j==state.patch_indices)[0].item()
            
            # (local) index of j's sym node
            rev_local_j = rev_node_lut[local_j]
            rev_global_j = state.patch_indices[rev_local_j]

            mask = ((state.edges[:,0] == rev_global_i) & (state.edges[:,1] == rev_global_j)) | ((state.edges[:,1] == rev_global_i) & (state.edges[:,0] == rev_global_j)) 
            assert mask.sum() == 1

            idx_l = np.arange(state.n_edges)[mask].item()

            rev_edge_lut[idx_k] = idx_l

        # External edge
        else:
            this_ext_idx = n_ext_indices[local_i]

            # Who knows why reverse edge indexing has to be reversed, but it really doesnt matter
            #.   since all external edges formed by node i are equivalent, anyway
            #if p % 2 == 0:
            idx_l = state.nodes_to_ext_edges[rev_local_i][::-1][this_ext_idx]
            #else:
                #idx_l = state.nodes_to_ext_edges[rev_local_i][this_ext_idx]

            rev_edge_lut[idx_k] = idx_l
            n_ext_indices[local_i] += 1

    assert np.array_equal(state.ext_count, n_ext_indices)

    labels = np.zeros(state.M_int+state.N_ext, dtype=int)
    labels[:] = -1

    i_label = 0
    # For each internal edge (indexed k)
    for local_k, k_edge_internal in enumerate(state.edges_int_indices):
        
        assert local_k == np.argwhere(state.edges_int_indices==k_edge_internal).item() 

        # Already dealt with this edge (thru symmetry)
        if labels[local_k] > -1:
            continue

        # Symmetrically equivalent edge l
        l_edge_internal = rev_edge_lut[k_edge_internal]

        local_l = np.argwhere(state.edges_int_indices==l_edge_internal).item()

        labels[local_k] = i_label
        labels[local_l] = i_label

        i_label += 1

    assert labels[:state.M_int].min() == 0
    assert labels[:state.M_int].max() == state.M_int // 2

    # Now peripheral nodes
    for i_local_ext, i_ext_idx in enumerate(state.ext_indices):

        this_idx = state.M_int+i_local_ext

        if labels[this_idx] > -1:
            continue
        
        # (local) patch index j of equivalent node to i
        j_ext_idx = rev_node_lut[i_ext_idx]

        j_local_ext = np.argwhere(state.ext_indices==j_ext_idx).item()

        labels[this_idx] = i_label
        labels[state.M_int+j_local_ext] = i_label

        i_label += 1


    return rev_node_lut, rev_edge_lut, labels

# From a list of labels for each symmetric edge group,
#   Expand to get one label per each edge
#.   This time, labels is shape (n_unique_sym_edge_groups,)
#
#
#.  labels: shape (n_unique_sym_edge_groups, ):   labels for each sym group after clustering
#.  sym_edge_labels: shape (n_edges, ): label for each edge according to its sym group (pre-merge)
#
#
#. Returns edge_labels (shape: (n_edges,))
def get_sym_edge_labels(labels, sym_edge_labels, state):
    n_edges = state.n_edges

    assert sym_edge_labels.size == n_edges
    assert np.unique(sym_edge_labels).size == labels.size

    final_edge_labels = np.zeros_like(sym_edge_labels)

    n_clust_final = np.unique(labels).size

    for i, final_label in enumerate(labels):
        # All the edges in this symmetry group
        mask = (sym_edge_labels == i)
        final_edge_labels[mask] = final_label

    return final_edge_labels

# Partition data into k groups
#def partition_feat(feat_vec, k=3)

### Merge edge types
k_cv=5
energies, ols_feat_vec, states = extract_from_ds('data/sam_pattern_04_04.npz')
feat_vec1 = get_feat_vec(states)

# Shape: M_tot+1 (hkoo for each edge, plus ko)
feat_vec2 = get_feat_vec_oo(states)
# shape: M_int+N_ext+1 (hkoo for each internal, number of external oo edges for each periph, plus ko)
feat_vec3 = test_get_feat_vec2(states)


state = states[np.argwhere(ols_feat_vec[:,0] == 0).item()]

# Full feature vector (edge type for each edge, 3*M_tot)
perf1, err1, _, _, reg1 = fit_k_fold(feat_vec1, energies, k=k_cv, do_ridge=True)
# h_oo (one for each of M_tot edges), plus k_o. Has redundancies (periph edges)
perf2, err2, _, _, reg2 = fit_k_fold(feat_vec2, energies, k=k_cv, do_ridge=True)
# ko, sum over all internal edges, sum over all external edges (miext edges for each peripheral node); shape: (M_int + N_periph + 1)
perf3, err3, _, _, reg3 = fit_k_fold(feat_vec3, energies, k=k_cv)
perf_m3, err_m3, _, _, reg_m3 = fit_k_fold(ols_feat_vec, energies, k=k_cv) 


rev_node_lut, rev_edge_lut, labels = find_sym_edges(state)

assert labels.min() == 0

# Shape: (n_edges,); gives symmetry group label for each edge
new_labels = merge_and_label(labels, state)
assert labels.max() == new_labels.max()

n_sym_group = np.unique(new_labels).size
sym_group_indices = np.arange(n_sym_group)

cmap = mpl.cm.tab10
norm = plt.Normalize(0,labels.max())

# Shape: (n_sym_edge_groups+1, ); number of oo edges for each edge sym group, plus ko
sym_feat_vec = construct_red_feat(feat_vec2, np.append(new_labels, new_labels.max()+1))

perf_sym, err_sym, _, _, reg_sym = fit_k_fold(sym_feat_vec, energies, k=k_cv)




# Greedily merge coefficients #
###############################

n_clust = 2

merge_reg = linear_model.LinearRegression()
this_labels = sym_group_indices.copy()

for i_round in range(0, sym_group_indices.size-n_clust):
    print("merge round {}".format(i_round))

    this_min_mse = np.inf
    this_min_labels = None
    this_min_reg = linear_model.LinearRegression()
    this_red_feat = None

    for idx in itertools.combinations(sym_group_indices, 2):
        # Groups already merged
        if this_labels[idx[0]] == this_labels[idx[1]]:
            continue

        # Indices of groups to merge
        merge_idx = np.array(idx)

        # These groups should not have been merged
        assert np.unique((this_labels[merge_idx])).size == 2

        mask = (this_labels == this_labels[merge_idx][0]) | (this_labels == this_labels[merge_idx][1])
        
        assert np.array_equal(np.intersect1d(sym_group_indices[mask], merge_idx), merge_idx)
        merge_idx = sym_group_indices[mask]

        # Indices of other groups
        other_idx = np.delete(sym_group_indices, merge_idx)
        
        # Indices of other groups, maintaining any merged groups
        other_grp_idx = []

        for l in np.unique(this_labels[other_idx]):
            mask = this_labels[other_idx] == l
            other_grp_idx.append(other_idx[mask])


        test_labels = this_labels.copy()
        # Label merged groups with 0's
        test_labels[merge_idx] = 0

        # Now re-label the remaining groups
        l = 1
        for other_grp in other_grp_idx:
            test_labels[other_grp] = l
            l += 1

        test_red_feat = construct_red_feat(sym_feat_vec, np.append(test_labels, test_labels.max()+1))
        merge_reg.fit(test_red_feat, energies)
        pred = merge_reg.predict(test_red_feat)
        this_mse = np.mean((energies-pred)**2)

        if this_mse < this_min_mse:
            this_min_mse = this_mse

            this_min_labels = test_labels.copy()
            this_min_reg.fit(test_red_feat, energies)

            this_red_feat = test_red_feat.copy()

        #print("merging: {}. mse: {:.8f}".format(merge_idx, this_mse))

    this_labels = this_min_labels.copy()

assert this_labels.max() == n_clust - 1

red_feat = construct_red_feat(sym_feat_vec, labels=np.append(this_labels, this_labels.max()+1))
perf_red, err_red, _, _, reg_red = fit_k_fold(red_feat, energies, k=k_cv)

final_edge_labels = get_sym_edge_labels(this_labels, new_labels, state)


'''
### CLUSTERING #######
#######################

## Cluster edge coefficients (leave ko coef alone)
n_clust = 2
clust = AgglomerativeClustering(n_clusters=n_clust, linkage='ward', affinity='euclidean')

coef = reg_sym.coef_[:-1].reshape(-1,1)
clust.fit(coef)

red_feat = construct_red_feat(sym_feat_vec, labels=np.append(clust.labels_, clust.labels_.max()+1))
perf_red, err_red, _, _, reg_red = fit_k_fold(red_feat, energies, k=k_cv)


final_edge_labels = get_sym_edge_labels(clust.labels_, new_labels, state)
'''

