import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib import *
from scratch.interactions.util import *
from scipy.spatial import cKDTree

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from scipy.spatial import cKDTree

import os, glob

homedir = os.environ['HOME']
plt.close('all')
def make_feat(methyl_mask, patch_indices):
    feat = np.zeros(pos_ext.shape[0])
    feat[patch_indices[methyl_mask]] = 1
    feat[patch_indices[~methyl_mask]] = -1

    return feat

def plot_feat(feat, ny=8, nz=8):
    this_feat = feat.reshape(ny, nz).T[::-1, :]

    return this_feat.reshape(1,1,ny,nz)

def get_edge_indices(nn_ext, patch_indices):
    edge_indices = []
    for i in range(36):
        neigh_idx = nn_ext[i]
        for j in neigh_idx:
            if j not in patch_indices:
                edge_indices.append(i)
                break

    return edge_indices

feat_vec, energies, poly, beta_phi_stars, positions, patch_indices, methyl_pos, adj_mat = load_and_prep()
n_dat = feat_vec.shape[0]

ds = np.load("sam_pattern_data.dat.npz", allow_pickle=True)

k_oh = 36 - methyl_pos.sum(axis=1)
idx = 700

pos_ext = gen_pos_grid(13, 13, z_offset=True, shift_y=-1, shift_z=-1)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

nn, nn_ext, _, _ = construct_neighbor_dist_lists(positions, pos_ext)

edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)

k_eff = np.zeros((n_dat, len(edges), 5))
# num hydroxyls on edge
n_edge = np.zeros(n_dat)
edge_indices = get_edge_indices(nn_ext, patch_indices)

for i, methyl_mask in enumerate(methyl_pos):
    this_k_eff = get_keff_all(methyl_mask, edges, patch_indices)
    k_eff[i] = this_k_eff
    n_edge[i] = (~methyl_mask[edge_indices]).sum()

# Shape: (n_dat, 5)
# n_mm n_oo n_mo n_me n_oe
k_eff = k_eff.sum(axis=1)



mynorm = plt.Normalize(-1,1)
plt.close('all')
idx = 693
feat = make_feat(methyl_pos[idx], patch_indices)
feat = plot_feat(feat)

plot_hextensor(feat, norm=mynorm)
plt.savefig("{}/Desktop/idx_{}.png".format(homedir, idx), transparent=True)

