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

from wang_landau import WangLandau


## Will be much faster if we cython this
def get_energy_tmp(pt_idx, methyl_mask, edges=None, patch_indices=None, reg=None):

    # n_mm_int, n_mo_int, n_mo_ext
    feat_vec = get_keff_all(methyl_mask, edges, patch_indices).sum(axis=0)[[0,2,3]]
    
    return np.dot(reg.coef_, feat_vec) + reg.intercept_

def get_energy(pt_idx, m_mask, nn, ext_count, coef1, coef2, coef3, inter):
    mm = 0
    mo_int = 0
    mo_ext = 0
    for m_idx in pt_idx:
        for n_idx in nn[m_idx]:
            if n_idx > m_idx:
                mm += m_mask[n_idx]
            mo_int += ~m_mask[n_idx]
        mo_ext += ext_count[m_idx]

    return inter + coef1*mm + coef2*mo_int + coef3*mo_ext

parser = argparse.ArgumentParser('Find entropy (density of states) of energy for a fixed k_ch3 \
                                  Using linear fit on n_mm, n_mo_int, and n_mo_ext')
parser.add_argument('--k-ch3', default=1, type=int,
                    help='k_ch3 to find density of states (default: %(default)s)')
parser.add_argument('--do-brute', action='store_true',
                    help='If true, get density of states by exhaustively generating each configuration at this \
                          k_ch3 (default: only do brute if k_ch3 <= 6 or k_ch3 >= 30')
parser.add_argument('--e-min', default=135.0, type=float,
                    help='Minimum energy, inclusive, in kT (default: %(default)s)')
parser.add_argument('--e-max', default=286.0, type=float,
                    help='Maximum energy, inclusive, in kT (default: %(default)s)')
parser.add_argument('--de', default=0.5, type=float,
                    help='Bin width for energy, in kT (default: %(default)s)')
parser.add_argument('--eps', type=float, default='1e-10',
                    help='Epsilon (tolerance) for Wang Landau (default: 1e-10)')
parser.add_argument('--hist-flat-tol', type=float, default=0.8,
                    help='Criterion for determining if histogram is flat for each \
                          Wang-Landau iteration (default: %(default)s)')

args = parser.parse_args()
k_ch3 = args.k_ch3
do_brute =  args.do_brute if args.do_brute else (k_ch3 <= 6 or k_ch3 >= 30)

e_min, e_max, de = args.e_min, args.e_max, args.de
bins = np.arange(e_min, e_max+de, de)

print("Doing k_ch3={:d},  do_brute={}".format(k_ch3, do_brute))
print("  with {:d} energy bins from {:0.1f} to {:0.1f}".format(bins.size-1, e_min, e_max))


## Load datasets to train model
ds = np.load('sam_pattern_data.dat.npz')

energies = ds['energies']
k_vals = ds['k_vals']
methyl_pos = ds['methyl_pos']
positions = ds['positions']

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
# patch_idx is list of patch indices in pos_ext 
#   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)
ext_indices = np.setdiff1d(np.arange(pos_ext.shape[0]), patch_indices)

# nn_ext is dictionary of (global) nearest neighbor's to each patch point
#   nn_ext[i]  global idxs of neighbor to local patch i 
nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)


# A count of external neighbor nodes for each (local) patch idx
ext_count = np.zeros(36, dtype=int)
for i in range(36):
    ext_count[i] = np.intersect1d(ext_indices, nn_ext[i]).size

# All methyl patterns
methyl_pos = ds['methyl_pos']
n_configs = methyl_pos.shape[0]

edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]
int_mask = np.ones(n_edges, dtype=bool)
int_mask[ext_indices] = False

k_eff_all_shape = np.load('k_eff_all.dat.npy')
k_eff_int = k_eff_all_shape[:, int_mask, :].sum(axis=1)
k_eff_ext = k_eff_all_shape[:,~int_mask, :].sum(axis=1)

# n_mm_int, n_mo_int, n_mo_ext
feat_vec = np.hstack((k_eff_int[:, [0,2]], k_eff_ext[:, 2][:, None]))
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)


# Run WL
fn_kwargs = dict(nn=nn, ext_count=ext_count, coef1=reg.coef_[0], coef2=reg.coef_[1], coef3=reg.coef_[2], inter=reg.intercept_)
wl = WangLandau(positions, bins, get_energy, fn_kwargs=fn_kwargs, eps=args.eps)

wl.gen_states(k_ch3, do_brute, hist_flat_tol=args.hist_flat_tol)

np.savez_compressed('density_k_c_{:d}.dat'.format(k_ch3), bins=bins, density=wl.density, k_ch3=k_ch3, eps=args.eps, hist_flat_tol=args.hist_flat_tol,
                    do_brute=do_brute, omega=wl.omega)
