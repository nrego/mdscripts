from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import networkx as nx
from scipy.spatial import cKDTree

from sklearn import datasets, linear_model

from scipy.integrate import cumtrapz

from util import gen_pos_grid, construct_neighbor_dist_lists, enumerate_edges
from util import plot_pattern, plot_3d, plot_graph, plot_annotate, plot_edges, plot_edge_list
from util import gen_w_graph
from util import find_keff, find_keff_kernel, get_keff_all
from util import fit_general_linear_model

import itertools

from sklearn.cluster import AgglomerativeClustering


def merge_grp(grp_i, merged_pts):
    ret_grps = np.array([])

    # Base case
    if grp_i.size == 1:
        return np.append(ret_grps, grp_i)
    else:
        for m_i in grp_i:
            ret_grps = np.append(ret_grps, merge_grp(merged_pts[m_i], merged_pts))


    return ret_grps

# Merge non-singleton clusters i and j, re-label all non-singleton
def merge_and_renumber_cluster(labels, idx_i, idx_j):
    n_non_singleton = np.unique(labels) - 1
    assert n_non_singleton > 1

    # Sorted current cluster labels
    clust_labels = np.unique(labels)[1:]
    assert np.array_equal(clust_labels, np.arange(n_non_singleton))

    # Clust labels for all other clusters (that we aren't merging)
    clust_labels = np.delete(labels, [idx_i, idx_j])

    new_labels = labels.copy()

    new_labels[(labels == idx_i) || (labels == idx_j)] = 0

    for idx, i in enumerate(clust_labels):
        new_labels[labels==i] = idx+1

    return new_labels


mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 30})

### run from pooled_pattern_sample directory, after linking in ~/mdscripts/scratch/sam/util/ ###

ds = np.load('sam_pattern_data.dat.npz')
energies = ds['energies']
# 2d positions (in y, z) of all 36 patch head groups (For plotting schematic images, calculating order params, etc)
# Shape: (N=36, 2)
positions = ds['positions']

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
# patc_idx is list of patch indices
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)


nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

methyl_pos = ds['methyl_pos']
n_configs = methyl_pos.shape[0]

edges = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]

## Sanity check - plot edges ##
assert n_edges == 131
plot_edge_list(pos_ext, edges, patch_indices, do_annotate=True)


## Find k_eff for each point - this each config now has a 131x5 length feature vector!
k_eff_all = np.zeros((n_configs, n_edges*5))

for i, methyl_mask in enumerate(methyl_pos):
    this_keff = get_keff_all(methyl_mask, edges, patch_indices)

    k_eff_all[i] = this_keff.ravel()


# Shape: (n_configs, 131, 5)
#   i.e. n_configs, n_edges, n_merge_grps
k_eff_all_shape = k_eff_all.reshape((n_configs, n_edges, 5))

# Compare to k_eff computed the other way (i.e. in fit_keff) - more sanity
k_eff_prev = np.loadtxt('k_eff_all.dat')

assert np.array_equal(k_eff_prev, k_eff_all_shape.sum(axis=1))


# Now remove oo and oe - redundant terms
#    Also treat n_me and n_mo equivalently - different edge types will automatically account for external
#    edges, if necessary
k_eff_shape = np.delete(k_eff_all_shape, (2,3), axis=2)
k_eff_shape = np.dstack((k_eff_shape[:,:,0][:,:,None], np.sum(k_eff_shape[:,:,1:], axis=2, keepdims=True)))
k_eff = k_eff_shape.reshape((n_configs, n_edges*2))

perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(k_eff, energies, do_ridge=True)
coefs = reg.coef_.reshape((n_edges, 2))

# distance matrix between coefs
tree = cKDTree(coefs)
d_l2 = tree.sparse_distance_matrix(tree, np.inf, p=2).toarray()
d_l1 = tree.sparse_distance_matrix(tree, np.inf, p=1).toarray()

## Now start merging groups?? 

# edge type lookup - for edge index i, merge_grp[i] = type_edge_i
# Initially each edge is its own type

clust = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='l1')
clust.fit(coefs)

#plot_edge_list(pos_ext, edges, patch_indices, annotation=clust.labels_)

clust_labels_all = np.zeros((clust.children_.shape[0], n_edges), dtype=int)


merged_pts = dict()
for i in range(n_edges):
    merged_pts[i] = np.array([i], dtype=int)

max_val = 0
for i_round, (m_i, m_j) in enumerate(clust.children_):

    #if m_i >= n_edges or m_j >= n_edges:
    #    break

    plt.close('all')
    grp_i = merge_grp(merged_pts[m_i], merged_pts)
    grp_j = merge_grp(merged_pts[m_j], merged_pts)

    # What kind of merge?

    # m_i and m_j are getting merged...
    this_merge = np.append(grp_i, grp_j)

    merged_pts[n_edges+i_round] = this_merge

# For each merging round, color clusters of edge types (non-clustered edge types remain black)
edge_types = np.ones(n_edges) * -1
n_unique_grps = 0

for i_round, (m_i, m_j) in enumerate(clust.children_):
    n_unique_groups = np.unique(edge_types).size - 1
    this_merge = merged_pts[n_edges+i_round]

    edge_types[this_merge.astype(int)] = n_unique_groups

    clust = AgglomerativeClustering(n_clusters=n_edges-i_round-1, linkage='average', affinity='l1').fit(coefs)

    plt.close('all')
    this_colors = np.zeros((n_edges, 4))
    this_colors[:,-1] = 1
    norm = plt.Normalize(0, edge_types.max())
    this_colors[edge_types != -1] = cm.tab10(norm(clust.labels_[edge_types != -1]))

    line_styles = []
    for idx in range(n_edges):
        sty = '-' if edge_types[idx] != -1 else '--'
        line_styles.append(sty)


    plot_edge_list(pos_ext, edges, patch_indices, do_annotate=False, colors=this_colors, line_styles=line_styles, ax=None)
    
    plt.savefig("/Users/nickrego/Desktop/merge_round_{}".format(i_round))



