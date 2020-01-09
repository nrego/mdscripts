from __future__ import division, print_function

import numpy as np
from scipy.special import binom 
from scipy.spatial import cKDTree
from itertools import combinations

import matplotlib as mpl

from matplotlib import pyplot as plt

from scipy.spatial import cKDTree
import os

mpl.rcParams.update({'axes.labelsize': 60})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':40})

from util import *

homedir = os.environ['HOME']

z_space = 0.5 # 0.5 nm spacing
y_space = np.sqrt(3)/2.0 * z_space
pos_ext = gen_pos_grid(ny=24, z_offset=True)
positions = gen_pos_grid(ny=12, z_offset=True, shift_y=6.0, shift_z=6)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)
edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]
int_indices = np.setdiff1d(np.arange(n_edges), ext_indices)

methyl_mask = np.ones_like(patch_indices, dtype=bool)
hydroxyl_mask = ~methyl_mask

k_eff_hydroxyl = get_keff_all(hydroxyl_mask, edges, patch_indices)
k_eff_oh_int = k_eff_hydroxyl[int_indices].sum(axis=0)
k_eff_oh_ext = k_eff_hydroxyl[ext_indices].sum(axis=0)
k_eff_methyl = get_keff_all(methyl_mask, edges, patch_indices)
k_eff_ch3_int = k_eff_methyl[int_indices].sum(axis=0)
k_eff_ch3_ext = k_eff_methyl[ext_indices].sum(axis=0)

m2_coef = np.array([-1.99519508, -0.91546819])
m2_int = 282.23667687264373

m3_coef = np.array([-1.66049927, -0.46594186, -0.17488402])
m3_int = 284.56628291300677

f_oh_act = np.loadtxt('/Users/nickrego/simulations/large_patch/umbr_k_000/PvN.dat')[0,1]
f_oh_err = np.loadtxt('/Users/nickrego/simulations/large_patch/umbr_k_000/PvN.dat')[0,2]
f_ch3_act = np.loadtxt('/Users/nickrego/simulations/large_patch/umbr_k_144/PvN.dat')[0,1]
f_ch3_err = np.loadtxt('/Users/nickrego/simulations/large_patch/umbr_k_144/PvN.dat')[0,2]

arr2 = np.array([positions.shape[0], k_eff_ch3_int[0]])
arr3 = np.array([k_eff_ch3_int[0], k_eff_ch3_int[2], k_eff_ch3_ext[3]])


print("actual difference: {:.2f}".format(f_ch3_act - f_oh_act))
print("M2 pred diff: {:.2f}".format(np.dot(m2_coef, arr2)))
print("M3 pred diff: {:.2f}".format(np.dot(m3_coef, arr3)))


