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

    reg = linear_model.LinearRegression()
    #reg = linear_model.Ridge(alpha=alpha)
    
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
        # slc is indices of validation (excluded from training) data set
        slc = slice(k*n_cohort, (k+1)*n_cohort)
        y_validate = y_rand[slc]
        X_validate = X_rand[slc]

        # Get training samples. np.delete makes a copy and **does not** act on array in-place
        y_train = np.delete(y_rand, slc)
        X_train = np.delete(X_rand, slc, axis=0)

        reg.fit(X_train, y_train)
        pred = reg.predict(X_validate)
        mse = np.mean((pred - y_validate)**2)
        perf_r2[k] = reg.score(X_validate, y_validate)
        perf_mse[k] = mse

    reg.fit(X, y)
    fit = reg.predict(xvals)

    pred = reg.predict(X)
    err = y - pred

    return(perf_r2, perf_mse, err, xvals, fit, reg)

def plot_3d(x, y, z, colors='k'):
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

def plot_annotate(positions, annotations, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(positions[:,0], positions[:,1], 'o', markeredgecolor='k', markeredgewidth=2, markerfacecolor='w', markersize=24)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for idx in range(36):
        ax.annotate(annotations[idx], xy=positions[idx]-0.025)

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


def gen_pos_grid(ny=6, nz=None, z_offset=False, shift_y=0, shift_z=0):
    if nz is None:
        nz = ny
    ## Generate grid of center points
    z_space = 0.5 # 0.5 nm spacing
    y_space = np.sqrt(3)/2.0 * z_space

    y_pos = 0 + shift_y*y_space
    pos_row = np.arange(0,0.5*(nz+1), 0.5) + shift_z*z_space

    positions = []
    for i in range(ny):
        if not z_offset:
            this_pos_row = pos_row if i % 2 == 0 else pos_row + z_space/2.0
        else:
            this_pos_row = pos_row if i % 2 != 0 else pos_row + z_space/2.0


        for j in range(nz):
            z_pos = this_pos_row[j]
            positions.append(np.array([y_pos, z_pos]))

        y_pos += y_space


    return np.array(positions)

def construct_neighbor_dist_lists(positions, pos_ext):
    ## Set up dict of nearest neighbors
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=0.51)

    tree_ext = cKDTree(pos_ext)
    ext_neighbors = tree.query_ball_tree(tree_ext, r=0.51)

    d_self, i_self = tree.query(positions, k=36)
    d_ext, i_ext = tree_ext.query(positions, k=108)

    # Dict of nearest neighbor patch indices
    #   for patch index i, nn[i] = {j}; j is index of patch atom that is nearest neighbor to patch atom i
    # Dict of nearest neighbor extended (non-patch) indices
    #    for patch index i, nn_ext[i] = {k}; k is index of extended pos (non-patch) atom that is nearest neighbor to patch atom i
    nn = dict()
    nn_ext = dict()
    for i in range(36):
        nn[i] = np.array([], dtype=int)
        nn_ext[i] = np.array(ext_neighbors[i], dtype=int)
    for i,j in pairs:
        assert j>i
        nn[i] = np.sort(np.append(nn[i], j))
        nn[j] = np.sort(np.append(nn[j], i))
        

    # Dict of patch i's distance to every other patch point - its distance to itself is set to infinity
    #  dd[i,k] is distance from point i to point k
    dd = np.zeros((36,36))
    dd_ext = np.zeros((36,108))
    for i in range(36):
        assert np.array_equal(np.sort(i_self[i]), np.arange(36))
        sort_idx_self = np.argsort(i_self[i])
        dd[i] = d_self[i][sort_idx_self] - 0.5
        dd[i,i] = np.inf

        assert np.array_equal(np.sort(i_ext[i]), np.arange(108))
        sort_idx_ext = np.argsort(i_ext[i])
        dd_ext[i] = d_ext[i][sort_idx_ext] - 0.5


    return nn, nn_ext, dd, dd_ext


def find_keff(methyl_mask, nn, nn_ext):

    deg = np.zeros((36,5))
    for i in range(36):
        nn_idx = nn[i]
        # All its extended (non-patch) neighbors are the same type, by definition
        nn_ext_mask = nn_ext[i].astype(bool)
        # is atom i a methyl?
        i_mask = methyl_mask[i]
        # are atom i's nearest patch neighbors methyls?
        nn_mask = methyl_mask[nn_idx]

        mm = (i_mask & nn_mask).sum()
        oo = (~i_mask & ~nn_mask).sum()
        mo = ((~i_mask & nn_mask) | (i_mask & ~nn_mask)).sum()
        # Methyl - extended 
        me = (i_mask & nn_ext_mask).sum()
        # Hydroxyl-extended
        oe = (~i_mask & nn_ext_mask).sum()

        #deg += mm*wt_mm + oo*wt_oo + mo*wt_mo


        deg[i] = [mm, oo, mo, me, oe]

    ## Sanity - every node's total degree should be 6
    degsum = deg.sum(axis=0)
    assert degsum[0] % 2 == 0
    assert degsum[1] % 2 == 0
    assert degsum[2] % 2 == 0

    assert (deg.sum(axis=1) == 6).all()

    deg[:,0] /= 2
    deg[:,1] /= 2
    deg[:,2] /= 2

    return deg

## gaussian kernal function
gaus = lambda x, lam: np.exp(-lam * x**2)

def find_keff_kernel(methyl_mask, dd, dd_ext, lam_mm=1, lam_oo=1, lam_mo=1, lam_me=1, lam_oe=1):
    
    deg = np.zeros((36,5))
    for i in range(36):
        # Distance of i to every other patch index
        d_self = dd[i]
        # Distance of i to every extended (non-patch) atom
        d_ext = dd_ext[i]

        # is atom i a methyl?
        i_mask = methyl_mask[i]
        
        # i-j connections (j on patch) that are m-m
        mm = (gaus(d_self[(i_mask & methyl_mask)], lam_mm)).sum()
        oo = (gaus(d_self[(~i_mask & ~methyl_mask)], lam_oo)).sum()

        mo_mask = ((~i_mask & methyl_mask) | (i_mask & ~methyl_mask))
        mo = (gaus(d_self[mo_mask], lam_mo)).sum()
        #connection to extended methyl-extended
        if i_mask:
            me = ( gaus(d_ext, lam_me) ).sum()
            oe = 0
        #connection to extended hydroxyl-extended
        else:
            me = 0
            oe = ( gaus(d_ext, lam_oe) ).sum()

        deg[i] = [mm, oo, mo, me, oe]

    # Double counted #
    deg[:,0] /= 2
    deg[:,1] /= 2
    deg[:,2] /= 2

    return deg







