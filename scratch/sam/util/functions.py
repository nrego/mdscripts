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

import math

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 30})

# regress y on set of n_dim features, X.
#   Note this can do a polynomial regression on a single feature -
#   just make each nth degree a power of that feature
def fit_general_linear_model(X, y, sort_axis=0, do_ridge=False, alpha=1, sample_weight=None, fit_intercept=True):
    np.random.seed()
    if sample_weight is None:
        sample_weight = np.ones_like(y)
    assert y.ndim == 1
    n_dat = y.size
    # Single feature (1d linear regression)
    if X.ndim == 1:
        X = X[:,None]
    
    # For plotting fit...
    sort_idx = np.argsort(X[:,sort_axis])
    xvals = X[sort_idx, :]

    if do_ridge:
        reg = linear_model.Ridge(alpha=alpha, fit_intercept=fit_intercept)
    else:
        reg = linear_model.LinearRegression(fit_intercept=fit_intercept)
    
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

        if sample_weight is None:
            reg.fit(X_train, y_train)
        else:
            train_weight = np.delete(sample_weight, slc)
            train_weight /= train_weight.sum()
            try:
                reg.fit(X_train, y_train, sample_weight=train_weight.copy())
            except:
                embed()

        pred = reg.predict(X_validate)
        if sample_weight is None:
            mse = np.mean((pred - y_validate)**2)
            perf_r2[k] = reg.score(X_validate, y_validate)
        else:
            valid_weight = sample_weight[slc].copy()
            valid_weight /= valid_weight.sum()
            mse = np.dot(valid_weight, (pred - y_validate)**2)
            perf_r2[k] = reg.score(X_validate, y_validate, sample_weight=valid_weight.copy())
        perf_mse[k] = mse

    reg.fit(X, y, sample_weight=sample_weight.copy())
    fit = reg.predict(xvals)

    pred = reg.predict(X)
    err = y - pred

    return (perf_r2, perf_mse, err, xvals, fit, reg)


# regress y on set of n_dim features, X.
#   Do leave-one-out CV
def fit_leave_one(X, y, sort_axis=0, fit_intercept=True, weights=None):

    assert y.ndim == 1
    n_dat = y.size
    # Single feature (1d linear regression)
    if X.ndim == 1:
        X = X[:,None]
    
    if weights is None:
        weights = np.ones_like(y)

    # For plotting fit...
    sort_idx = np.argsort(X[:,sort_axis])
    xvals = X[sort_idx, sort_axis]
    if xvals.min() > 0:
        xvals = np.append(0, xvals)

    reg = linear_model.LinearRegression(fit_intercept=fit_intercept)

    # R^2 and MSE for each train/validation round
    perf_mse = np.zeros(n_dat)


    # Choose one of the cohorts as validation set, train on remainder.
    #   repeat for each cohort
    for k in range(n_dat):

        # Get training samples. np.delete makes a copy and **does not** act on array in-place
        y_train = np.delete(y, k)
        X_train = np.delete(X, k, axis=0)
        w_train = np.delete(weights, k)

        y_validate = y[k]
        X_validate = X[k].reshape(1,-1)

        reg.fit(X_train, y_train, sample_weight=w_train)
        pred = reg.predict(X_validate)

        mse = ((y_validate - pred)**2).item()
        perf_mse[k] = mse

    reg.fit(X, y, sample_weight=weights)
    fit = reg.predict(xvals.reshape(-1,1))

    pred = reg.predict(X)
    err = y - pred

    return (perf_mse, err, xvals, fit, reg)

# Perform bootstrapping on data to estimate errors in linear regression coefficients
def fit_bootstrap(X, y, fit_intercept=True, n_bootstrap=1000, weights=None):

    np.random.seed()
    assert y.ndim == 1
    n_dat = y.size
    # Single feature (1d linear regression)
    if X.ndim == 1:
        X = X[:,None]

    if weights is None:
        weights = np.ones_like(y)
    
    # For plotting fit...
    boot_inter = np.zeros(n_bootstrap)
    boot_coef = np.zeros((n_bootstrap, X.shape[1]))

    reg = linear_model.LinearRegression(fit_intercept=fit_intercept)

    for i_boot in range(n_bootstrap):
        dat_indices = np.random.choice(n_dat, size=n_dat, replace=True)

        this_boot_y = y[dat_indices]
        this_boot_X = X[dat_indices, ...]
        this_boot_w = weights[dat_indices]

        reg.fit(this_boot_X, this_boot_y, sample_weight=this_boot_w)

        boot_inter[i_boot] = reg.intercept_
        boot_coef[i_boot] = reg.coef_ 

    return (boot_inter, boot_coef)

def plot_3d(x, y, z, **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z, **kwargs)

    
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
    #plt.show()

    return ax

# Draw edge types (including to outside)
def plot_edges(positions, methyl_mask, pos_ext, nn, nn_ext, ax=None):
    edge_in = ['b--', 'k--', 'k-']
    edge_out = ['b:', 'k:']
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    n = positions.shape[0]
    assert len(nn.keys()) == len(nn_ext.keys()) == n

    for i in range(n):
        meth_i = int(methyl_mask[i])
        neighbor_idx_in = nn[i]
        neighbor_idx_out = nn_ext[i]

        for j in neighbor_idx_in:
            if j < i:
                continue
            meth_j = int(methyl_mask[j])

            # 0: oo, 1: mo, 2: mm
            edge_type_in = meth_i+meth_j

            ax.plot([positions[i,0], positions[j,0]], [positions[i,1], positions[j,1]], edge_in[edge_type_in])
            
        for j in neighbor_idx_out:
            ax.plot([positions[i,0], pos_ext[j,0]], [positions[i,1], pos_ext[j,1]], edge_out[meth_i])


    return ax

# Generates a list of all edges 
def enumerate_edges(positions, pos_ext, nn_ext, patch_indices):
    # nn_ext's are indexed by local patch index -> give global index of nn
    assert len(nn_ext.keys()) == positions.shape[0]

    edges = []
    # Indices of edges to external OH's
    ext_indices = []
    for i in range(positions.shape[0]):
        # Index of this patch point in pos_ext
        global_i = patch_indices[i]
        neighbor_idx = nn_ext[i]
        for j in neighbor_idx:
            # patch-patch edge that's already been seen
            if j in patch_indices and j <= global_i:
                continue

            # This is an external edge, so save this edge's index to ext_indices
            if j not in patch_indices:
                ext_indices.append(len(edges))

            edges.append((global_i,j))

    edges = np.array(edges)

    return edges, np.array(ext_indices)

def plot_edge_list(pos_ext, edges, patch_indices, do_annotate=True, annotation=None, colors=None, line_styles=None, line_widths=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if annotation is None:
        annotation = np.arange(edges.shape[0])


    for i_edge, (i,j) in enumerate(edges):

        i_int = i in patch_indices
        j_int = j in patch_indices

        i_symbol = 'ko' if i_int else 'rx'
        j_symbol = 'ko' if j_int else 'rx'

        if line_styles is None:
            edge_style = '-' #if (i_int and j_int) else '--'
        else:
            edge_style = line_styles[i_edge]

        if line_widths is None:
            line_width = 3
        else:
            line_width = line_widths[i_edge]

        ax.plot(pos_ext[i,0], pos_ext[i,1], i_symbol, markersize=12, zorder=3)
        ax.plot(pos_ext[j,0], pos_ext[j,1], j_symbol, markersize=12, zorder=3)

        if colors is not None:
            this_color = colors[i_edge]
        else:
            this_color = 'k'
        ax.plot([pos_ext[i,0], pos_ext[j,0]], [pos_ext[i,1], pos_ext[j,1]], color=this_color, linestyle=edge_style, linewidth=line_width)

        midpt = (pos_ext[i] + pos_ext[j]) / 2.0

        if do_annotate:
            ax.annotate(annotation[i_edge], xy=midpt-0.025)


def plot_annotate(positions, annotations, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(positions[:,0], positions[:,1], 'o', markeredgecolor='k', markeredgewidth=2, markerfacecolor='w', markersize=24)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for idx, note in enumerate(annotations):
        ax.annotate(note, xy=positions[idx]-0.025)

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

    n_positions = positions.shape[0]
    n_pos_ext = pos_ext.shape[0]
    ## Set up dict of nearest neighbors
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=0.51)

    tree_ext = cKDTree(pos_ext)
    ext_neighbors = tree.query_ball_tree(tree_ext, r=0.51)

    d_self, i_self = tree.query(positions, k=n_positions)
    d_ext, i_ext = tree_ext.query(positions, k=n_pos_ext)

    # Dict of nearest neighbor patch indices
    #   for patch index i, nn[i] = {j}; j is index of patch atom that is nearest neighbor to patch atom i
    # Dict of nearest neighbor extended (non-patch) indices
    #    for patch index i, nn_ext[i] = {k}; k is index of extended pos (non-patch) atom that is nearest neighbor to patch atom i
    nn = dict()
    nn_ext = dict()
    for i in range(n_positions):
        nn[i] = np.array([], dtype=int)
        nn_ext[i] = np.array(ext_neighbors[i], dtype=int)
    for i,j in pairs:
        assert j>i
        nn[i] = np.sort(np.append(nn[i], j))
        nn[j] = np.sort(np.append(nn[j], i))
        

    # Dict of patch i's distance to every other patch point - its distance to itself is set to infinity
    #  dd[i,k] is distance from point i to point k
    dd = np.zeros((n_positions,n_positions))
    dd_ext = np.zeros((n_positions,n_pos_ext))
    for i in range(n_positions):
        try:
            assert np.array_equal(np.sort(i_self[i]), np.arange(n_positions))
            sort_idx_self = np.argsort(i_self[i])
            dd[i] = d_self[i][sort_idx_self] #- 0.5
        except:
            pass
        dd[i,i] = np.inf

        assert np.array_equal(np.sort(i_ext[i]), np.arange(n_pos_ext))
        sort_idx_ext = np.argsort(i_ext[i])
        dd_ext[i] = d_ext[i][sort_idx_ext] #- 0.5


    return nn, nn_ext, dd, dd_ext


def aic_ols(reg, err):
    n_sample = err.size
    sse = np.sum(err**2)
    n_param = reg.coef_.size + 1

    return n_sample * np.log(sse/n_sample) + 2*n_param

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

# Slightly different inputs, but should give same as find_keff, only now for each of the 131 edges
def get_keff_all(methyl_mask, edges, patch_indices):

    # each line is: n_mm, n_oo, n_mo, n_me, n_oe
    deg = np.zeros((edges.shape[0], 5), dtype=float)

    for i_edge, (i,j) in enumerate(edges):

        global_methyl_indices = patch_indices[methyl_mask]
        # Are i,j in patch or external?
        i_patch = i in patch_indices
        j_patch = j in patch_indices

        # At least one must be in patch
        assert (i_patch or j_patch)

        # Is this an internal edge type?
        int_edge = i_patch and j_patch

        # Is i, j a methyl?
        i_meth = i in global_methyl_indices
        j_meth = j in global_methyl_indices

        # Flags
        mo_flag = (i_meth and (j_patch and not j_meth)) or (j_meth and (i_patch and not i_meth))
        me_flag = (i_meth and not j_patch) or (j_meth and not i_patch)
        oe_flag = (not i_meth and not j_patch) or (not j_meth and not i_patch)

        i_bin = np.array([i_meth, not i_meth and i_patch, mo_flag, me_flag, oe_flag])
        j_bin = np.array([j_meth, not j_meth and j_patch, mo_flag, me_flag, oe_flag])

        deg[i_edge] = (i_bin & j_bin)

    assert deg.sum() == edges.shape[0]

    return deg

## gaussian kernal function
gaus = lambda x, sig_sq: np.exp(-x**2/(2*sig_sq))

## Truncated, shifted gaussian
def gaus_cut(x, sig_sq, xcut):
    x = np.array(x, ndmin=1)
    g = np.exp(-x**2/(2*sig_sq)) - np.exp(-xcut**2/(2*sig_sq))
    pref = 1 / ( np.sqrt(np.pi*2*sig_sq) * math.erf(xcut/np.sqrt(2*sig_sq)) - 2*xcut*np.exp(-xcut**2/2*sig_sq) )
    g[x**2>xcut**2] = 0

    return pref*g


def find_keff_kernel(methyl_mask, dd, dd_ext, sig_sq, rcut):

    sig_sq_mm, sig_sq_oo, sig_sq_mo, sig_sq_me, sig_sq_oe = sig_sq
    rcut_mm, rcut_oo, rcut_mo, rcut_me, rcut_oe = rcut
    
    deg = np.zeros((36,5))
    for i in range(36):
        # Distance of i to every other patch index
        d_self = dd[i]
        # Distance of i to every extended (non-patch) atom
        d_ext = dd_ext[i]

        # is atom i a methyl?
        i_mask = methyl_mask[i]
        
        # i-j connections (j on patch) that are m-m
        mm = (gaus_cut(d_self[(i_mask & methyl_mask)], sig_sq_mm, rcut_mm)).sum()
        oo = (gaus_cut(d_self[(~i_mask & ~methyl_mask)], sig_sq_oo, rcut_oo)).sum()

        mo_mask = ((~i_mask & methyl_mask) | (i_mask & ~methyl_mask))
        mo = (gaus_cut(d_self[mo_mask], sig_sq_mo, rcut_mo)).sum()
        #connection to extended methyl-extended
        if i_mask:
            me = ( gaus_cut(d_ext, sig_sq_me, rcut_me) ).sum()
            oe = 0
        #connection to extended hydroxyl-extended
        else:
            me = 0
            oe = ( gaus_cut(d_ext, sig_sq_oe, rcut_oe) ).sum()

        deg[i] = [mm, oo, mo, me, oe]

    # Double counted #
    deg[:,0] /= 2
    deg[:,1] /= 2
    deg[:,2] /= 2

    return deg


def extract_data(fname="sam_pattern_data.dat.npz"):
    ds = np.load(fname)

    energies = ds['energies']
    k_vals = ds['k_vals']
    methyl_pos = ds['methyl_pos']
    positions = ds['positions']

    # All methyl patterns
    methyl_pos = ds['methyl_pos']
    n_configs = methyl_pos.shape[0]

    # extended grid so we can find patch indices on edge of patch
    pos_ext = gen_pos_grid(8, z_offset=True, shift_y=-1, shift_z=-1)

    # patch_indices is a list of the (global) indices of points on pos_ext corresponding
    #   to the patch points on positions
    #
    #   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
    d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

    # nn_ext is dictionary of (global) nearest neighbors to each (local) patch point
    #   nn_ext: [local position i] => [list of global pos_ext nearest neighbors]
    #   nn_ext[i]  global idxs of neighbor to local patch i 
    nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

    # edges is a list of (i,j) tuples of (global) indices
    #   only contains nn edges between either patch-patch or patch-external
    # ext_indices contains indices of all patch-external edges
    edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
    n_edges = edges.shape[0]

    int_indices = np.setdiff1d(np.arange(n_edges), ext_indices)


    return (energies, methyl_pos, k_vals, positions, pos_ext, patch_indices, nn, nn_ext, edges, ext_indices, int_indices)

