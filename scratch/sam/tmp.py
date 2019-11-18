import os, glob
import numpy as np
from scratch.sam.util import *

fnames = sorted(glob.glob('*.dat'))

def find_ext_edge(positions, pos_ext):
    d, patch_indices = cKDTree(pos_ext).query(positions, k=1)
    nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

    int_edges = dict()
    ext_edges = dict()
    for i in range(positions.shape[0]):
        ext_edges[i] = 0
        int_edges[i] = nn[i].size

        for j in nn_ext[i]:
            if j not in patch_indices:
                ext_edges[i] += 1


    ext_nodes = (np.array(list(ext_edges.values())) > 0).sum()

    return int_edges, ext_edges, ext_nodes

ds = np.load('sam_pattern_data.dat.npz')

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)

pos_66 = gen_pos_grid(6)
pos_44 = gen_pos_grid(4)
pos_22 = gen_pos_grid(2)
pos_23 = gen_pos_grid(2,3)
pos_49 = gen_pos_grid(4, 9, shift_z=-1.5, shift_y=1)

int_edges66, ext_edges66, n_ext66 = find_ext_edge(pos_66, pos_ext)
int_edges44, ext_edges44, n_ext44 = find_ext_edge(pos_44, pos_ext)
int_edges22, ext_edges22, n_ext22 = find_ext_edge(pos_22, pos_ext)
int_edges49, ext_edges49, n_ext49 = find_ext_edge(pos_49, pos_ext)
int_edges23, ext_edges23, n_ext23 = find_ext_edge(pos_23, pos_ext)


feat = np.array([[2,2], [2,3], [4,4], [6,6], [4,9]])
ext = np.array([np.sum(list(ext_edges22.values())), np.sum(list(ext_edges23.values())), np.sum(list(ext_edges44.values())), np.sum(list(ext_edges66.values())), np.sum(list(ext_edges49.values()))])
n_ext = np.array([n_ext22, n_ext23, n_ext44, n_ext66, n_ext49])

