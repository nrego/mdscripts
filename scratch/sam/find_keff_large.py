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