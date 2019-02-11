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


def plot_pattern(positions, methyl_mask, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    ax.set_xticks([])
    ax.set_yticks([])
    pos = positions[methyl_mask]
    ax.plot(positions[:,0], positions[:,1], 'bo', markersize=18)
    ax.plot(pos[:,0], pos[:,1], 'ko', markersize=18)
    plt.show()


# Pos is a collection of points, shape (n_pts, ndim)
def get_rms(pos, prec=2):
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_diff = np.linalg.norm(diff, axis=1)**2
    return np.round( np.sqrt(sq_diff.mean()), prec)

def gen_graph(positions, indices):

    if indices.size == 0:
        return nx.Graph()

    if indices.ndim == 0:
        indices = indices[None]

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

def gen_w_graph(positions, methyl_mask):
    indices_all = np.arange(36)
    indices_ch3 = indices_all[methyl_mask]
    indices_oh = indices_all[~methyl_mask]

    tree = cKDTree(positions)
    edges = list(tree.query_pairs(r=0.6))

    graph = nx.Graph()
    node_dict = [(idx, dict(phob=methyl_mask[idx])) for idx in indices_all]
    graph.add_nodes_from(node_dict)
    
    for i,j in edges:
        if i in indices_ch3 and j in indices_ch3:
            weight = 1
        elif i in indices_oh and j in indices_oh:
            weight = 0
        else:
            weight = 0.5

        graph.add_edge(i,j,weight=weight)


    return graph

# Template for analyzing mu_ex for all patch patterns
homedir = os.environ['HOME']
ds_k = np.load('pattern_sample/analysis_data.dat.npz')
ds_l = np.load('inv_pattern_sample/analysis_data.dat.npz')

positions = ds_k['positions']
assert np.array_equal(positions, ds_l['positions'])

rms_bins = ds_k['rms_bins']
assert np.array_equal(rms_bins, ds_l['rms_bins'])

k_bins = ds_k['k_bins']
assert np.array_equal(k_bins, ds_l['k_bins'])

xx, yy = np.meshgrid(k_bins[:-1], rms_bins[:-1])
k_vals_unpack = l_vals = xx.ravel()
rms_vals_unpack = yy.ravel()

# Now unpack and combine all data...
dat_k = ds_k['energies'].ravel()
mask_k = ~np.ma.masked_invalid(dat_k).mask
dat_k = dat_k[mask_k]
k_vals_k = k_vals_unpack[mask_k]
rms_vals_k = []
n_clust_k = []
edge_sum_k = []
methyl_pos_k = ds_k['methyl_mask'].reshape(xx.size, 36)[mask_k,:]
for pos_mask in methyl_pos_k:
    this_pos = positions[pos_mask]
    indices = np.arange(36)[pos_mask]
    graph = gen_graph(positions, indices)
    n_clust_ch3 = len(list(nx.connected_components(graph)))
    n_clust_k.append(n_clust_ch3)
    if pos_mask.sum() == 0:
        this_rms_val = 0
    else:
        this_rms_val = get_rms(this_pos)
    rms_vals_k.append(this_rms_val)
    w_graph = gen_w_graph(positions, pos_mask)
    esum = np.array([d['weight'] for (u,v,d) in w_graph.edges(data=True)]).sum()
    edge_sum_k.append(esum)

rms_vals_k = np.array(rms_vals_k)
n_clust_k = np.array(n_clust_k)
edge_sum_k = np.array(edge_sum_k)

dat_l = ds_l['energies'].ravel()
mask_l = ~np.ma.masked_invalid(dat_l).mask
dat_l = dat_l[mask_l]
k_vals_l = 36 - k_vals_unpack[mask_l]
methyl_pos_l = ds_l['methyl_mask'].reshape(xx.size, 36)[mask_l,:]
rms_vals_l = []
n_clust_l = []
edge_sum_l = []
# Need to get the D_ch3 values from positions...
for pos_mask in methyl_pos_l:
    this_pos = positions[pos_mask]
    if pos_mask.sum() == 0:
        this_rms_val = 0
    else:
        this_rms_val = get_rms(this_pos)
    rms_vals_l.append(this_rms_val)
    indices = np.arange(36)[pos_mask]
    graph = gen_graph(positions, indices)
    n_clust_ch3 = len(list(nx.connected_components(graph)))
    n_clust_l.append(n_clust_ch3)
    w_graph = gen_w_graph(positions, pos_mask)
    esum = np.array([d['weight'] for (u,v,d) in w_graph.edges(data=True)]).sum()
    edge_sum_l.append(esum)
edge_sum_l = np.array(edge_sum_l)
rms_vals_l = np.array(rms_vals_l)
n_clust_l = np.array(n_clust_l)

dat_pooled = np.append(dat_k, dat_l)
k_vals = np.append(k_vals_k, k_vals_l)
rms_vals = np.append(rms_vals_k, rms_vals_l)
n_clust = np.append(n_clust_k, n_clust_l)
methyl_pos = np.vstack((methyl_pos_k, methyl_pos_l))
edge_sum = np.append(edge_sum_k, edge_sum_l)

w_graph = gen_w_graph(positions, methyl_pos[500])
m_nodes = [n for (n,d) in w_graph.nodes(data=True) if d['phob']]
o_nodes = [n for (n,d) in w_graph.nodes(data=True) if not d['phob']]
esmall = [(u,v) for (u,v,d) in w_graph.edges(data=True) if d['weight'] == 0 ]
emed = [(u,v) for (u,v,d) in w_graph.edges(data=True) if d['weight'] == 0.5 ]
elarge = [(u,v) for (u,v,d) in w_graph.edges(data=True) if d['weight'] == 1 ]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xticks([])
ax.set_yticks([])

nx.draw_networkx_nodes(w_graph, pos=positions, nodelist=m_nodes, node_color='k', ax=ax)
nx.draw_networkx_nodes(w_graph, pos=positions, nodelist=o_nodes, node_color='b', ax=ax)
nx.draw_networkx_edges(w_graph, pos=positions, edgelist=esmall, edge_color='b', style='dotted', ax=ax)
nx.draw_networkx_edges(w_graph, pos=positions, edgelist=emed, edge_color='k', style='dashed', ax=ax)
nx.draw_networkx_edges(w_graph, pos=positions, edgelist=elarge, edge_color='k', style='solid', ax=ax)

