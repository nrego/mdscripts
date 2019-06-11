from __future__ import division, print_function

import os, glob
from scipy.spatial import cKDTree
from util import *


dirnames = np.sort(glob.glob('../*pattern_sample/*/d_*/trial_0'))
n_dat = dirnames.size + 2 # for k=0 and k=36

old_ds = np.load('old_sam_pattern_data.dat.npz')
positions = old_ds['positions']

methyl_base = np.zeros(36, dtype=bool)

methyl_pos = np.zeros((n_dat, 36), dtype=bool)
k_vals = np.zeros(n_dat)
energies = np.zeros(n_dat)

for i, dirname in enumerate(dirnames):
    methyl_mask = methyl_base.copy()
    pt_pos = np.loadtxt('{}/this_pt.dat'.format(dirname), dtype=int)
    methyl_mask[pt_pos] = True
    k_ch3 = methyl_mask.sum()

    energy = np.loadtxt('{}/PvN.dat'.format(dirname))[0,1]

    methyl_pos[i] = methyl_mask
    k_vals[i] = k_ch3
    energies[i] = energy

# k_00
energy = np.loadtxt('../pattern_sample/k_00/PvN.dat')[0,1]
energies[-2] = energy
k_vals[-2] = 0

# k_36
energy = np.loadtxt('../pattern_sample/k_36/f_k_all.dat')[-1]
energies[-1] = energy
k_vals[-1] = 36
methyl_pos[-1][:] = True

np.savez_compressed('sam_pattern_data.dat', energies=energies, positions=positions, k_vals=k_vals, methyl_pos=methyl_pos)

## Find k_eff_all - enumerate all edge types
pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)
# patch_idx is list of patch indices in pos_ext 
#   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
n_edges = edges.shape[0]

n_samples = energies.shape[0]

# shape: (n_samples, n_edges, 5)
#   where edge types are:
#   n_mm, n_oo, n_mo, n_me, n_oe
k_eff_all_shape = np.zeros((n_samples, n_edges, 5))

for i, methyl_mask in enumerate(methyl_pos):
    k_eff_all_shape[i, ...] = get_keff_all(methyl_mask, edges, patch_indices)

# n_oo + n_oe
oo = k_eff_all_shape[:,:,1] + k_eff_all_shape[:,:,4]
# n_mo + n_me
mo = k_eff_all_shape[:,:,2] + k_eff_all_shape[:,:,3]
mm = k_eff_all_shape[:,:,0]


# Now only mm, oo, mo;  still have redundancy
k_eff_all_shape = np.dstack((mm, oo, mo))

np.save('k_eff_all.dat', k_eff_all_shape)