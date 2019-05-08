import os, glob
from scipy import special
from util import *


pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
coef = np.array([-1.66049927, -0.46594186, -0.17488402])

pred = []
for n in range(2,7):
    positions = gen_pos_grid(n)

    # patch_idx is list of patch indices in pos_ext 
    #   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
    d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

    # nn_ext is dictionary of (global) nearest neighbor's to each patch point
    #   nn_ext[i]  global idxs of neighbor to local patch i 
    nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

    edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
    n_edges = edges.shape[0]
    int_indices = np.setdiff1d(np.arange(n_edges), ext_indices)

    methyl_mask = np.ones(n*n, dtype=bool)
    k_eff_all = get_keff_all(methyl_mask, edges, patch_indices)

    int_edges = k_eff_all[int_indices].sum(axis=0)
    ext_edges = k_eff_all[ext_indices].sum(axis=0)

    n_mm_int = int_edges[0]
    n_mo_int = int_edges[2]
    n_mo_ext = ext_edges[-2]

    delta_g = np.dot(coef, np.array([n_mm_int, n_mo_int, n_mo_ext]))
    pred.append(delta_g)

    print("n: {:01d}   dg: {:0.2f}".format(n, delta_g))

pred = np.array(pred)