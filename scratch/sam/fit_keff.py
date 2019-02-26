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

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 30})

# regress y on set of n_dim features, X.
#   Note this can do a polynomial regression on a single feature -
#   just make each nth degree a power of that feature
def fit_general_linear_model(X, y, sort_axis=0, alpha=1):
    np.random.seed()

    assert y.ndim == 1
    n_dat = y.size
    # Single feature (1d linear regression)
    if X.ndim == 1:
        X = X[:,None]
    
    # For plotting fit...
    sort_idx = np.argsort(X[:,sort_axis])
    xvals = X[sort_idx, :]

    #reg = linear_model.LinearRegression()
    reg = linear_model.Ridge(alpha=alpha)
    
    # Randomly split data into fifths
    n_cohort = n_dat // 5
    rand_idx = np.random.permutation(n_dat)
    # y_rand, X_rand are just the shuffled energies and predictors
    y_rand = y[rand_idx]
    X_rand = X[rand_idx]

    # R^2 and MSE for each train/validation round
    perf_r2 = np.zeros(5)
    perf_mse = np.zeros(5)

    # Choose one of the cohorts as validation set, train on remainder.
    #   repeat for each cohort
    for k in range(5):
        slc = slice(k*n_cohort, (k+1)*n_cohort)
        y_validate = y_rand[slc]
        X_validate = X_rand[slc]

        y_train = np.delete(y_rand, slc)
        X_train = np.delete(X_rand, slc, axis=0)

        reg.fit(X_train, y_train)
        pred = reg.predict(X_validate)
        mse = np.mean((pred - y_validate)**2)
        perf_r2[k] = reg.score(X_validate, y_validate)
        perf_mse[k] = mse

    reg.fit(X, y)
    fit = reg.predict(xvals)

    return(perf_r2, perf_mse, xvals, fit, reg)

def plot_3d(x, y, z, colors):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, c=colors)

    
def plot_graph(w_graph, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    m_nodes = [n for (n,d) in w_graph.nodes(data=True) if d['phob']]
    o_nodes = [n for (n,d) in w_graph.nodes(data=True) if not d['phob']]
    esmall = [(u,v) for (u,v,d) in w_graph.edges(data=True) if d['weight'] == -1 ]
    emed = [(u,v) for (u,v,d) in w_graph.edges(data=True) if d['weight'] == -0.5 ]
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

def gen_w_graph(positions, methyl_mask, wt_mm=1, wt_oo=-1, wt_mo=-0.5):
    indices_all = np.arange(positions.shape[0])
    indices_ch3 = indices_all[methyl_mask]
    indices_oh = indices_all[~methyl_mask]

    tree = cKDTree(positions)
    edges = list(tree.query_pairs(r=0.6))

    graph = nx.Graph()
    node_dict = [(idx, dict(phob=methyl_mask[idx])) for idx in indices_all]
    graph.add_nodes_from(node_dict)
    
    for i,j in edges:
        if i in indices_ch3 and j in indices_ch3:
            weight = wt_mm
        elif i in indices_oh and j in indices_oh:
            weight = wt_oo
        else:
            weight = wt_mo

        graph.add_edge(i,j,weight=weight)

    return graph

def find_keff(nn, methyl_mask):

    deg = np.zeros(3)
    for i in range(36):
        nn_idx = nn[i]
        i_mask = methyl_mask[i]
        nn_mask = methyl_mask[nn_idx]

        mm = (i_mask & nn_mask).sum()
        oo = (~i_mask & ~nn_mask).sum()
        mo = ((~i_mask & nn_mask) | (i_mask & ~nn_mask)).sum()

        #deg += mm*wt_mm + oo*wt_oo + mo*wt_mo
        deg += [mm, oo, mo]

    return deg



### run ###

ds = np.load('pooled_pattern_sample/sam_pattern_data.dat.npz')

# 2d positions (in y, z) of all 36 patch head groups (For plotting schematic images, calculating order params, etc)
# Shape: (N=36, 2)
positions = ds['positions']

# array of masks (boolean array) of methyl groups for each sample configuration
#  Shape: (n_samples=884, N=36)
methyl_pos = ds['methyl_pos']

# k_ch3 for each sample
k_vals = ds['k_vals']

# \beta F(0)'s
energies = ds['energies']

## Set up dict of nearest neighbors
tree = cKDTree(positions)
pairs = tree.query_pairs(r=0.51)

nn = dict()
for i in range(36):
    nn[i] = np.array([], dtype=int)
for i,j in pairs:
    assert j>i
    nn[i] = np.sort(np.append(nn[i], j))
    nn[j] = np.sort(np.append(nn[j], i))