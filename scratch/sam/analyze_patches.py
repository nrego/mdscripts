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

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 30})

# regress y on set of n_dim features, X.
#   Note this can do a polynomial regression on a single feature -
#   just make each nth degree a power of that feature
def fit_general_linear_model(X, y):
    # Single feature (1d linear regression)
    if X.ndim == 1:
        X = X[:,None]
        
    reg = linear_model.LinearRegression()
    reg.fit(X, y)

def plot_3d(x, y, z, colors):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, c=colors)

    fig.show()

def unpack_data(ds):
    energies = ds['energies'].ravel()
    size = energies.size
    mask = ~np.ma.masked_invalid(energies).mask
    energies = energies[mask]
    methyl_pos = ds['methyl_mask'].reshape(size, 36)
    methyl_pos = methyl_pos[mask]

    return (energies, methyl_pos)

def plot_graph(w_graph, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    m_nodes = [n for (n,d) in w_graph.nodes(data=True) if d['phob']]
    o_nodes = [n for (n,d) in w_graph.nodes(data=True) if not d['phob']]
    esmall = [(u,v) for (u,v,d) in w_graph.edges(data=True) if d['weight'] == 0 ]
    emed = [(u,v) for (u,v,d) in w_graph.edges(data=True) if d['weight'] == 0.5 ]
    elarge = [(u,v) for (u,v,d) in w_graph.edges(data=True) if d['weight'] == 1 ]

    ax.set_xticks([])
    ax.set_yticks([])

    nx.draw_networkx_nodes(w_graph, pos=positions, nodelist=m_nodes, node_color='k', ax=ax)
    nx.draw_networkx_nodes(w_graph, pos=positions, nodelist=o_nodes, node_color='b', ax=ax)
    nx.draw_networkx_edges(w_graph, pos=positions, edgelist=esmall, edge_color='b', style='dotted', ax=ax)
    nx.draw_networkx_edges(w_graph, pos=positions, edgelist=emed, edge_color='k', style='dashed', ax=ax)
    nx.draw_networkx_edges(w_graph, pos=positions, edgelist=elarge, edge_color='k', style='solid', ax=ax)

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
    if pos.size == 0:
        return 0.0
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
        else:
            weight = 0

        graph.add_edge(i,j,weight=weight)


    return graph

# Template for analyzing energies_ex for all patch patterns
homedir = os.environ['HOME']
ds_k = np.load('pattern_sample/analysis_data.dat.npz')
ds_l = np.load('inv_pattern_sample/analysis_data.dat.npz')

## Just some constants stored in each array (should be the same in each)
positions = ds_k['positions']
assert np.array_equal(positions, ds_l['positions'])

rms_bins = ds_k['rms_bins']
assert np.array_equal(rms_bins, ds_l['rms_bins'])

k_bins = ds_k['k_bins']
assert np.array_equal(k_bins, ds_l['k_bins'])

extent = (rms_bins[0], rms_bins[-1], k_bins[0], k_bins[-1])

xx, yy = np.meshgrid(k_bins[:-1], rms_bins[:-1])
k_vals_unpack = l_vals = xx.ravel()
rms_vals_unpack = yy.ravel()

## Extract all values from k dataset
energies_k, methyl_pos_k = unpack_data(ds_k)
energies_l, methyl_pos_l = unpack_data(ds_l)

rms_k = np.zeros_like(energies_k)
for idx, methyl_mask in enumerate(methyl_pos_k):
    ch3_pos = positions[methyl_mask]

    rms_k[idx] = get_rms(ch3_pos)

# Pool all data 
energies = np.append(energies_k, energies_l)
methyl_pos = np.vstack((methyl_pos_k, methyl_pos_l))

# number of methyls
k_vals = methyl_pos.sum(axis=1)

# Now go through each configuration and generate whatever other data we might want
rms_ch3 = np.zeros_like(energies)
rms_oh = np.zeros_like(energies)
sum_edges = np.zeros_like(energies)
k_eff = np.zeros_like(energies)
sum_nodes = np.zeros_like(energies)
for idx, methyl_mask in enumerate(methyl_pos):
    ch3_pos = positions[methyl_mask]
    oh_pos = positions[~methyl_mask]

    rms_ch3[idx] = get_rms(ch3_pos)
    rms_oh[idx] = get_rms(oh_pos)

    w_graph = gen_w_graph(positions, methyl_mask)

    sum_edges[idx] = np.array([d['weight'] for (u,v,d) in w_graph.edges(data=True)]).sum()
    deg = np.array(dict(w_graph.degree(weight='weight')).values())
    k_eff[idx] = (deg == 6).sum()
    sum_nodes[idx] = deg.sum()

w_graph = gen_w_graph(positions, methyl_pos[500])
deg = np.array(dict(w_graph.degree(weight='weight')).values())

fig, ax = plt.subplots(figsize=(5.5,6))
ax.set_xticks([])
ax.set_yticks([])
ax.plot(positions[:,0], positions[:,1], 'o', markeredgecolor='k', markeredgewidth=2, markerfacecolor='w', markersize=24)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for idx in range(36):
    ax.annotate(deg[idx], xy=positions[idx]-0.025)
fig.savefig('positions_labeled.pdf')
plt.close('all')

min_energy = energies.min()
max_energy = energies.max()
norm = mpl.colors.Normalize(min_energy, max_energy)
cmap = cm.nipy_spectral
colors = cmap(norm(energies))
