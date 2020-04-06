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

from scratch.sam.util import *

import itertools

from sklearn.cluster import AgglomerativeClustering

homedir = os.environ['HOME']

plot_it = False

def gen_merged_keff(k_eff_shape, labels):
    
    n_pts = k_eff_shape.shape[0]
    # i.e. methyl-methyl, methyl-hydroxyl, etc
    n_edge_type = k_eff_shape.shape[2]

    n_unique_edges = np.unique(labels[labels!=-1]).size + (labels==-1).sum()

    #single, unique vectors
    singletons = k_eff_shape[:, labels==-1, :]
    k_eff_merge = singletons
    for i_label in np.unique(labels[labels!=-1]):
        k_eff_merge_i = k_eff_shape[:, labels==i_label, :].sum(axis=1)
        k_eff_merge = np.hstack((k_eff_merge, k_eff_merge_i[:,None,:]))


    #embed()
    return k_eff_merge.reshape((n_pts, n_unique_edges*n_edge_type))


def merge_grp(grp_i, merged_pts):
    ret_grps = np.array([], dtype=int)

    # Base case
    if grp_i.size == 1:
        return np.append(ret_grps, grp_i)
    else:
        for m_i in grp_i:
            ret_grps = np.append(ret_grps, merge_grp(merged_pts[m_i], merged_pts))


    return ret_grps

# Merge non-singleton clusters i and j, re-label all non-singleton
#
# How can we change the labels as little as possible??
def merge_and_renumber_cluster(labels, label_i, label_j):

    n_non_singleton = np.unique(labels[labels!=-1]).size


    # Can't be merging if we don't have at least two non-singleton clusters    
    assert n_non_singleton > 1

    # Sorted current cluster labels
    clust_labels = np.unique(labels[labels!=-1])
    assert np.array_equal(clust_labels, np.arange(n_non_singleton))

    # Clust labels for all other clusters (that we aren't merging)
    clust_labels = np.delete(clust_labels, [label_i, label_j])

    new_labels = labels.copy()


    #new_labels[(labels == label_i) | (labels == label_j)] = 0

    #for idx, i in enumerate(clust_labels):
    #    new_labels[labels==i] = idx+1

    # Re-label in a way that modifies existing labels as little as possible

    # Easy case: Either label_i or label_j is n_non_singleton-1
    #   in which case just set the label to the other
    if label_i == n_non_singleton - 1:
        new_labels[labels==label_i] = label_j
    elif label_j == n_non_singleton - 1:
        new_labels[labels==label_j] = label_i


    # More common case: label_i, label_j are not (n_non_singleton-1)
    #  Here, just set labels to which ever is lower; the labels at n_non_singleton-1
    #    are set to the other
    else:
        if label_i > label_j:
            new_labels[labels==label_i] = label_j
            new_labels[labels==n_non_singleton-1] = label_i
        else:
            new_labels[labels==label_j] = label_i
            new_labels[labels==n_non_singleton-1] = label_j      

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

edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]

## Sanity check - plot edges ##
assert n_edges == 131

fig, ax = plt.subplots(figsize=(6,6))
plot_edge_list(pos_ext, edges, patch_indices, do_annotate=True, ax=ax)
ax.axis('off')
plt.savefig('{}/Desktop/edges_labeled.png'.format(homedir))
plt.close('all')

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

#assert np.array_equal(k_eff_prev, k_eff_all_shape.sum(axis=1))


# n_mm  n_oo  n_mo  n_me   n_oe
#   0    1     2     3      4

# n_oo + n_oe
oo = k_eff_all_shape[:,:,1] + k_eff_all_shape[:,:,4]
# n_mo + n_me
mo = k_eff_all_shape[:,:,2] + k_eff_all_shape[:,:,3]
mm = k_eff_all_shape[:,:,0]


# Now only mm, oo, mo;  still have redundancy
k_eff_all_shape = np.dstack((mm, oo, mo))
n_connection_type = k_eff_all_shape.shape[2]
k_eff_all = k_eff_all_shape.reshape((n_configs, n_edges*n_connection_type))

perf_mse, err, xvals, fit, reg = fit_leave_one(k_eff_all, energies)
coefs = reg.coef_.reshape((n_edges, n_connection_type))

# distance matrix between coefs
tree = cKDTree(coefs)
d_l2 = tree.sparse_distance_matrix(tree, np.inf, p=2).toarray()
d_l1 = tree.sparse_distance_matrix(tree, np.inf, p=1).toarray()

## Now start merging groups?? 

# edge type lookup - for edge index i, merge_grp[i] = type_edge_i
# Initially each edge is its own type

clust = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='euclidean')
clust.fit(coefs)

#plot_edge_list(pos_ext, edges, patch_indices, annotation=clust.labels_)

clust_labels_all = np.zeros((clust.children_.shape[0], n_edges), dtype=int)
clust_labels_curr = np.zeros((n_edges,), dtype=int)
clust_labels_curr[:] = -1

# Contains list of all edges that are merged at this step (i=n_edges+i_step)
merged_pts = dict()
for i in range(n_edges):
    merged_pts[i] = np.array([i], dtype=int)

# Number of non singleton clusters
n_non_singleton = 0
n_singleton = n_edges
norm = plt.Normalize(0, 20)

mses = np.zeros(clust.children_.shape[0])
tot_mses = np.zeros_like(mses)
aics = np.zeros_like(mses)
# Iterate thru each step of hierarchical clustering
for i_round, (m_i, m_j) in enumerate(clust.children_):
    print("round: {}".format(i_round))


    assert n_singleton == (clust_labels_curr==-1).sum()
    assert np.unique(clust_labels_curr[clust_labels_curr!=-1]).size == n_non_singleton
    print("n non singleton clusters: {}".format(n_non_singleton))

    plt.close('all')
    grp_i = merge_grp(merged_pts[m_i], merged_pts)
    grp_j = merge_grp(merged_pts[m_j], merged_pts)

    # m_i and m_j are getting merged...
    this_merge = np.append(grp_i, grp_j)
    merged_pts[n_edges+i_round] = this_merge

    # What kind of merge?

    # singleton-singleton merge
    if grp_i.size == 1 and grp_j.size == 1:
        #embed()
        clust_labels_curr[this_merge] = n_non_singleton
        n_non_singleton += 1
        n_singleton -= 2

    # singleton-cluster merge
    elif grp_i.size == 1 or grp_j.size == 1:
        #embed()
        if grp_i.size > 1:
            clust_labels_curr[this_merge] = clust_labels_curr[grp_i][0]
        elif grp_j.size > 1:
            clust_labels_curr[this_merge] = clust_labels_curr[grp_j][0]
        else:
            raise ValueError
        n_singleton -= 1

    # Cluster-cluster merge
    else:
        label_i = clust_labels_curr[grp_i][0]
        label_j = clust_labels_curr[grp_j][0]
        clust_labels_curr = merge_and_renumber_cluster(clust_labels_curr, label_i, label_j)
        n_non_singleton -= 1


    clust_labels_all[i_round] = clust_labels_curr
    this_colors = np.zeros((n_edges, 4))
    this_colors[:,-1] = 1
    

    this_colors[clust_labels_curr!=-1] = cm.tab20(norm(clust_labels_curr[clust_labels_curr!=-1]))

    line_styles = []
    for label in clust_labels_curr:
        sty = '-' if label != -1 else '--'
        line_styles.append(sty)

    line_widths = np.ones(n_edges) * 3
    line_widths[this_merge] = 6

    if plot_it:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6,7))
        n_edge_type = np.unique(clust_labels_curr[clust_labels_curr!=-1]).size + (clust_labels_curr==-1).sum()
        plot_edge_list(pos_ext, edges, patch_indices, do_annotate=False, colors=this_colors, line_styles=line_styles, line_widths=line_widths, ax=ax)
        ax.axis('off')
        ax.set_title('Edges: {} \n Clusters: {}'.format(n_edge_type, n_non_singleton))
        fig.tight_layout()
        plt.savefig("{}/Desktop/merge_round_{:03d}".format(homedir, i_round))
        

    # Fit new model to merged struct
    merged_keff = gen_merged_keff(k_eff_all_shape, clust_labels_curr)
    perf_mse, err, xvals, fit, reg = fit_leave_one(merged_keff, energies)

    print("  N features: {}".format(merged_keff.shape[1]))
    print("  MSE, fit: {}".format(np.mean(err**2)))
    print("  MSE, CV: {}".format(perf_mse.mean()))

    mses[i_round] = perf_mse.mean()
    tot_mses[i_round] = np.mean(err**2)
    aics[i_round] = aic_ols(reg, err)


## Now remove redundancies
# Remove redundant features

# K_eff_all_shape:   shape: (n_samples, n_edges, n_conn_type)
# Conn types:
#             mm    oo    mo
#              0     1     2
# Note: we also have k_vals (# of methyls, also a function of connection types)
k_vals = ds['k_vals']
k_eff_shape = np.delete(k_eff_all_shape, 1, axis=2)
n_connection_type = k_eff_shape.shape[2]
k_eff = k_eff_shape.reshape((n_configs, n_edges*n_connection_type))





