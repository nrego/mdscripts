from __future__ import division, print_function

import os, glob
from scipy import special
from util import *

import matplotlib as mpl



ds = np.load('sam_pattern_data.dat.npz')
pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)

positions = gen_pos_grid(6, z_offset=False)
all_indices = np.arange(36).astype(int)

for i in range(6, 1, -1):
    patch_positions = gen_pos_grid(i)
    nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(patch_positions, pos_ext)
    d, patch_indices = cKDTree(pos_ext).query(patch_positions, k=1)
    edges, ext_indices = enumerate_edges(patch_positions, pos_ext, nn_ext, patch_indices)
    n_mo = len(edges[ext_indices])
    n_mm = len(edges) - n_mo
    dg = -2.8 * i**2 - 0.73 * n_mm + 0.29 * n_mo
    dg_ch3 = np.loadtxt('N_{}/ch3/f_k_all.dat'.format(i))[-1]
    dg_oh = np.loadtxt('N_{}/oh/f_k_all.dat'.format(i))[-1]

    print("N : {}".format(i))
    print("  dg pred: {:.2f} ".format(dg))
    print("  dg act: {:.2f}".format(dg_ch3 - dg_oh))