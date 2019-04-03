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

from util import *

import itertools

from sklearn.cluster import AgglomerativeClustering

def fit_model(data_vecs, indices, energies):
    reshaped_data = []
    for data_vec, idx in zip(data_vecs, indices):
        assert data_vec.ndim == 2
        new_vec = data_vec[:,idx]
        if new_vec.ndim == 1:
            reshaped_data.append(new_vec[:,None])
        elif new_vec.ndim == 2:
            reshaped_data.append(new_vec.copy())
        else:
            raise ValueError

    feat_vec = np.hstack(reshaped_data)

    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

    return reg

aics = []
perf_mses = []

## Compare models of different levels of sophistication - # params ##

ds = np.load('sam_pattern_data.dat.npz')

energies = ds['energies']
k_vals = ds['k_vals']
k_vals_both = np.hstack((k_vals[:,None], 36-k_vals[:,None]))
methyl_pos = ds['methyl_pos']
positions = ds['positions']

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
# patch_idx is list of patch indices
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

methyl_pos = ds['methyl_pos']
n_configs = methyl_pos.shape[0]

edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]

# shape: (n_samples, n_edges, n_conn_type)

# Conn types:   mm  oo  mo
k_eff_all_shape = np.load('k_eff_all.dat.npy')
assert n_edges == k_eff_all_shape.shape[1]


n_conn_type = k_eff_all_shape.shape[2]
## One model ##

# k_ch3 #
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(k_vals, energies, do_ridge=False)

# k_O #
perf_r2, perf_mse, err_orig, xvals, fit, reg = fit_general_linear_model(36-k_vals, energies, do_ridge=False)

aics.append(aic_ols(reg, err))
perf_mses.append(perf_mse.mean())

## n_mm n_oo  n_mo ##
conn_names = ['mm', 'oo', 'mo']

# Choose 2 to keep
k_eff_one_edge = k_eff_all_shape.sum(axis=1)
indices = list(itertools.combinations(np.arange(n_conn_type), 2))
print("2 model")
for idx in indices:

    ## two edges ##
    
    feat_vec = k_eff_all_shape[:,:,idx].sum(axis=1)
    conn_0 = conn_names[idx[0]]
    conn_1 = conn_names[idx[1]]
    
    print("Model: {} {}".format(conn_0, conn_1))
    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)
    reg2 = fit_model([k_eff_one_edge], [idx], energies)
    assert np.array_equal(reg.coef_, reg2.coef_)
    assert np.array_equal(reg.intercept_, reg2.intercept_)
    print("  alpha_0: {:0.2f}".format(reg.intercept_))
    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[0]))
    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[1]))
    print("  perf: {:0.2f}".format(perf_mse.mean()))

    ## k_C and edge
    
    idx_incl = np.setdiff1d(np.arange(n_conn_type), np.array(idx))[0]
    feat_vec = np.delete(k_eff_all_shape, idx, axis=2).sum(axis=1)
    feat_vec = np.hstack((k_vals[:,None], feat_vec))
    conn_0 = conn_names[idx_incl]
    print("Model: {} {}".format("k_C", conn_0))
    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

    print("  alpha_0: {:0.2f}".format(reg.intercept_))
    print("  alpha_kC: {:0.2f}".format(reg.coef_[0]))
    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
    print("  perf: {:0.2f}".format(perf_mse.mean()))

    # k_O and edge
    
    idx_incl = np.setdiff1d(np.arange(n_conn_type), np.array(idx))[0]
    feat_vec = np.delete(k_eff_all_shape, idx, axis=2).sum(axis=1)
    feat_vec = np.hstack((36-k_vals[:,None], feat_vec))
    conn_0 = conn_names[idx_incl]
    print("Model: {} {}".format("k_O", conn_0))
    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

    print("  alpha_0: {:0.2f}".format(reg.intercept_))
    print("  alpha_kO: {:0.2f}".format(reg.coef_[0]))
    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
    print("  perf: {:0.2f}".format(perf_mse.mean()))
aics.append(aic_ols(reg, err))
perf_mses.append(perf_mse.mean())

## 3 model ##
clust = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='euclidean')
k_eff_all = k_eff_all_shape.reshape((n_configs, n_edges*n_conn_type))
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(k_eff_all, energies, do_ridge=True)
coefs = reg.coef_.reshape((n_edges, n_conn_type))
clust.fit(coefs)

int_mask = clust.labels_ == 0
assert int_mask.sum() == 85

k_eff_int = k_eff_all_shape[:,int_mask, :].sum(axis=1)
k_eff_ext = k_eff_all_shape[:,~int_mask, :].sum(axis=1)
assert k_eff_ext[:,0].max() == 0

## n_mm n_oo  n_mo ##
conn_names_int = ['mm_int', 'oo_int', 'mo_int']
conn_names_ring = ['mm_rg', 'oo_rg', 'mo_rg']
conn_names_ext = ['mm_ext', 'oo_ext', 'mo_ext']
indices_int = list(itertools.combinations(np.arange(n_conn_type), 2))
indices_ring = list(itertools.combinations(np.arange(n_conn_type), 2))
indices_ext = [1,2]

print("\n\n")
print("3 model")
n_combos = 0
# k_idx=0: no k
# k_idx=1: k_C
# k_idx=2: k_O
meth_val = 133
oh_val = 286

meth_centric = []
oh_centric = []
tol = 6
for k_idx in [0,1,2]:

    ## No k case
    ## Chose two internal edges and one external ##
    if k_idx == 0:
        for int_idx in indices_int:
            
            for ext_idx in indices_ext:

                conn_0 = conn_names_int[int_idx[0]]
                conn_1 = conn_names_int[int_idx[1]]
                conn_2 = conn_names_ext[ext_idx]
                print("Model: {} {} {}".format(conn_0, conn_1, conn_2,))
                feat_vec = np.hstack((k_eff_int[:,int_idx], k_eff_ext[:,ext_idx][:,None]))
                perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)
                reg2 = fit_model([k_eff_int, k_eff_ext], [int_idx, ext_idx], energies)
                assert np.array_equal(reg.coef_, reg2.coef_)
                assert np.array_equal(reg.intercept_, reg2.intercept_)
                n_combos += 1
                if abs(reg.intercept_ - oh_val) < tol:
                    oh_centric.append([k_idx, int_idx, ext_idx])
                    print("  alpha_0: {:0.2f}".format(reg.intercept_))
                    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[0]))
                    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[1]))
                    print("  alpha_{}: {:0.2f}".format(conn_2, reg.coef_[2]))
                    print("  perf: {:0.2f}".format(perf_mse.mean()))
                elif abs(reg.intercept_ - meth_val) < tol:
                    meth_centric.append([k_idx, int_idx, ext_idx])
                    oh_centric.append([k_idx, int_idx, ext_idx])
                    print("  alpha_0: {:0.2f}".format(reg.intercept_))
                    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[0]))
                    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[1]))
                    print("  alpha_{}: {:0.2f}".format(conn_2, reg.coef_[2]))
                    print("  perf: {:0.2f}".format(perf_mse.mean()))
    else:
        if k_idx == 1:
            k_name='kC'
            k = k_vals
        else:
            k_name='kO'
            k = 36 - k_vals

        # Exclude an internal edge
        for int_incl in range(3):
            for ext_idx in indices_ext:

                conn_0 = conn_names_int[int_incl]
                conn_1 = conn_names_ext[ext_idx]
                print("Model: {} {} {}".format(k_name, conn_0, conn_1))
                feat_vec = np.hstack((k[:,None], k_eff_int[:,int_incl][:,None], k_eff_ext[:,ext_idx][:,None]))
                perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)
                reg2 = fit_model([k_vals_both, k_eff_int, k_eff_ext], [k_idx-1, int_incl, ext_idx], energies)
                assert np.array_equal(reg.coef_, reg2.coef_)
                assert np.array_equal(reg.intercept_, reg2.intercept_)

                n_combos += 1
                if abs(reg.intercept_ - oh_val) < tol:
                    oh_centric.append([k_idx, int_incl, ext_idx])
                    print("  alpha_0: {:0.2f}".format(reg.intercept_))
                    print("  alpha_{}: {:0.2f}".format(k_name, reg.coef_[0]))
                    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
                    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[2]))
                    print("  perf: {:0.2f}".format(perf_mse.mean()))
                elif abs(reg.intercept_ - meth_val) < tol:
                    meth_centric.append([k_idx, int_incl, ext_idx])
                    print("  alpha_0: {:0.2f}".format(reg.intercept_))
                    print("  alpha_{}: {:0.2f}".format(k_name, reg.coef_[0]))
                    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
                    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[2]))
                    print("  perf: {:0.2f}".format(perf_mse.mean()))
        # Exclude external edge
        for int_idx in indices_int:
            conn_0 = conn_names_int[int_idx[0]]
            conn_1 = conn_names_int[int_idx[1]]
            print("Model: {} {} {}".format(k_name, conn_0, conn_1))
            feat_vec = np.hstack((k[:,None], k_eff_int[:,int_idx]))
            perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)
            reg2 = fit_model([k_vals_both, k_eff_int], [k_idx-1, int_idx], energies)
            assert np.array_equal(reg.coef_, reg2.coef_)
            assert np.array_equal(reg.intercept_, reg2.intercept_)

            n_combos += 1
            if abs(reg.intercept_ - oh_val) < tol:
                oh_centric.append([k_idx, int_idx, ()])
                print("  alpha_0: {:0.2f}".format(reg.intercept_))
                print("  alpha_{}: {:0.2f}".format(k_name, reg.coef_[0]))
                print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
                print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[2]))
                print("  perf: {:0.2f}".format(perf_mse.mean()))
            elif abs(reg.intercept_ - meth_val) < tol:
                meth_centric.append([k_idx, int_idx, ()])
                print("  alpha_0: {:0.2f}".format(reg.intercept_))
                print("  alpha_{}: {:0.2f}".format(k_name, reg.coef_[0]))
                print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
                print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[2]))
                print("  perf: {:0.2f}".format(perf_mse.mean()))
aics.append(aic_ols(reg, err))
perf_mses.append(perf_mse.mean())

## Final three edge types ##
############################

clust = AgglomerativeClustering(n_clusters=3, linkage='average', affinity='euclidean')
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(k_eff_all, energies, do_ridge=True)
coefs = reg.coef_.reshape((n_edges, n_conn_type))
clust.fit(coefs)

addl_int_indices = np.array([14,25,60,103,106], dtype=int)
# Internal edges
int_mask = clust.labels_ == 0
int_mask[addl_int_indices] = True
assert int_mask.sum() == 63

# External edges
ext_mask = clust.labels_ == 1
assert ext_mask.sum() == 46

# Ring edges
ring_mask = clust.labels_ == 2
ring_mask[addl_int_indices] = False
assert ring_mask.sum() == 22

labels = clust.labels_.copy()
labels[int_mask] = 2
labels[ring_mask] = 1
labels[ext_mask] = 0

cmap = cm.tab20
colors = cmap(labels)

fig, ax = plt.subplots(figsize=(6,6))
plot_edge_list(pos_ext, edges, patch_indices, colors=colors, do_annotate=False, ax=ax)
ax.axis('off')
fig.savefig('/Users/nickrego/Desktop/fig_3.png')
plt.close('all')

k_eff_int = k_eff_all_shape[:,int_mask, :].sum(axis=1)
k_eff_ext = k_eff_all_shape[:,ext_mask, :].sum(axis=1)
assert k_eff_ext[:,0].max() == 0
k_eff_ring = k_eff_all_shape[:,ring_mask,:].sum(axis=1)

## n_mm n_oo  n_mo ##
conn_names_int = ['mm_int', 'oo_int', 'mo_int']
conn_names_ring = ['mm_rg', 'oo_rg', 'mo_rg']
conn_names_ext = ['mm_ext', 'oo_ext', 'mo_ext']
indices_int = list(itertools.combinations(np.arange(n_conn_type), 2))
indices_ring = list(itertools.combinations(np.arange(n_conn_type), 2))
indices_ext = [1,2]

print("\n\n")
print("5 model")
n_combos = 0
# k_idx=0: no k
# k_idx=1: k_C
# k_idx=2: k_O
for k_idx in [0,1,2]:

    ## No k case
    ## Chose two internal edges, two ring edges, and one external ##
    if k_idx == 0:
        for int_idx in indices_int:
            for ring_idx in indices_ring:
                for ext_idx in indices_ext:

                    conn_0 = conn_names_int[int_idx[0]]
                    conn_1 = conn_names_int[int_idx[1]]
                    conn_2 = conn_names_ring[ring_idx[0]]
                    conn_3 = conn_names_ring[ring_idx[1]]
                    conn_4 = conn_names_ext[ext_idx]
                    print("Model: {} {} {} {} {}".format(conn_0, conn_1, conn_2, conn_3, conn_4))
                    feat_vec = np.hstack((k_eff_int[:,int_idx], k_eff_ring[:,ring_idx], k_eff_ext[:,ext_idx][:,None]))
                    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

                    print("  alpha_0: {:0.2f}".format(reg.intercept_))
                    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[0]))
                    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[1]))
                    print("  alpha_{}: {:0.2f}".format(conn_2, reg.coef_[2]))
                    print("  alpha_{}: {:0.2f}".format(conn_3, reg.coef_[3]))
                    print("  alpha_{}: {:0.2f}".format(conn_4, reg.coef_[4]))
                    print("  perf: {:0.2f}".format(perf_mse.mean()))
                    n_combos += 1
                    reg2 = fit_model([k_eff_int, k_eff_ring, k_eff_ext], [int_idx, ring_idx, ext_idx], energies)
                    assert np.array_equal(reg.coef_, reg2.coef_)
                    assert np.array_equal(reg.intercept_, reg2.intercept_)
    else:
        if k_idx == 1:
            k_name='kC'
            k = k_vals
        else:
            k_name='kO'
            k = 36 - k_vals

        # Exclude an internal edge
        for int_incl in range(3):
            for ring_idx in indices_ring:
                for ext_idx in indices_ext:

                    conn_0 = conn_names_int[int_incl]
                    conn_1 = conn_names_ring[ring_idx[0]]
                    conn_2 = conn_names_ring[ring_idx[1]]
                    conn_3 = conn_names_ext[ext_idx]
                    print("Model: {} {} {} {} {}".format(k_name, conn_0, conn_1, conn_2, conn_3))
                    feat_vec = np.hstack((k[:,None], k_eff_int[:,int_incl][:,None], k_eff_ring[:,ring_idx], k_eff_ext[:,ext_idx][:,None]))
                    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

                    print("  alpha_0: {:0.2f}".format(reg.intercept_))
                    print("  alpha_{}: {:0.2f}".format(k_name, reg.coef_[0]))
                    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
                    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[2]))
                    print("  alpha_{}: {:0.2f}".format(conn_2, reg.coef_[3]))
                    print("  alpha_{}: {:0.2f}".format(conn_3, reg.coef_[4]))
                    print("  perf: {:0.2f}".format(perf_mse.mean()))
                    n_combos += 1
                    reg2 = fit_model([k_vals_both, k_eff_int, k_eff_ring, k_eff_ext], [k_idx-1, int_incl, ring_idx, ext_idx], energies)
                    assert np.array_equal(reg.coef_, reg2.coef_)
                    assert np.array_equal(reg.intercept_, reg2.intercept_)
        # Exclude a ring edge
        for int_idx in indices_int:
            for ring_incl in range(3):
                for ext_idx in indices_ext:

                    conn_0 = conn_names_int[int_idx[0]]
                    conn_1 = conn_names_int[int_idx[1]]
                    conn_2 = conn_names_ring[ring_incl]
                    conn_3 = conn_names_ext[ext_idx]

                    print("Model: {} {} {} {} {}".format(k_name, conn_0, conn_1, conn_2, conn_3))
                    feat_vec = np.hstack((k[:,None], k_eff_int[:,int_idx], k_eff_ring[:,ring_incl][:,None], k_eff_ext[:,ext_idx][:,None]))
                    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

                    print("  alpha_0: {:0.2f}".format(reg.intercept_))
                    print("  alpha_{}: {:0.2f}".format(k_name, reg.coef_[0]))
                    print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
                    print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[2]))
                    print("  alpha_{}: {:0.2f}".format(conn_2, reg.coef_[3]))
                    print("  alpha_{}: {:0.2f}".format(conn_3, reg.coef_[4]))
                    print("  perf: {:0.2f}".format(perf_mse.mean()))
                    n_combos += 1
                    reg2 = fit_model([k_vals_both, k_eff_int, k_eff_ring, k_eff_ext], [k_idx-1, int_idx, ring_incl, ext_idx], energies)
                    assert np.array_equal(reg.coef_, reg2.coef_)
                    assert np.array_equal(reg.intercept_, reg2.intercept_)
        # Exclude external edge
        for int_idx in indices_int:
            for ring_idx in indices_ring:
                conn_0 = conn_names_int[int_idx[0]]
                conn_1 = conn_names_int[int_idx[1]]
                conn_2 = conn_names_ring[ring_idx[0]]
                conn_3 = conn_names_ring[ring_idx[1]]
                print("Model: {} {} {} {} {}".format(k_name, conn_0, conn_1, conn_2, conn_3))
                feat_vec = np.hstack((k[:,None], k_eff_int[:,int_idx], k_eff_ring[:,ring_idx]))
                perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)

                print("  alpha_0: {:0.2f}".format(reg.intercept_))
                print("  alpha_{}: {:0.2f}".format(k_name, reg.coef_[0]))
                print("  alpha_{}: {:0.2f}".format(conn_0, reg.coef_[1]))
                print("  alpha_{}: {:0.2f}".format(conn_1, reg.coef_[2]))
                print("  alpha_{}: {:0.2f}".format(conn_2, reg.coef_[3]))
                print("  alpha_{}: {:0.2f}".format(conn_3, reg.coef_[4]))
                print("  perf: {:0.2f}".format(perf_mse.mean()))
                n_combos += 1
                reg2 = fit_model([k_vals_both, k_eff_int, k_eff_ring], [k_idx-1, int_idx, ring_idx], energies)
                assert np.array_equal(reg.coef_, reg2.coef_)
                assert np.array_equal(reg.intercept_, reg2.intercept_)
aics.append(aic_ols(reg, err))
perf_mses.append(perf_mse.mean())
