## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

import numpy as np
import pickle

from scratch.sam.util import gen_pos_grid
from scipy.spatial import cKDTree

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


## Miscellaneous helper functions ##

colors = [(0,0,1), (0,0,0)]
mymap = LinearSegmentedColormap.from_list('mymap', colors, N=100)

do_cnn = False
no_run = False

def plot_pattern(pos_ext, patch_indices, methyl_mask):
    pos = pos_ext[patch_indices]
    plt.plot(pos_ext[:,0], pos_ext[:,1], 'bo')
    plt.plot(pos[methyl_mask, 0], pos[methyl_mask, 1], 'ko')

    #plt.show()

# Plot a feature from a feature vector corresponding to a list of points
#   Plots each point colored according its feature
def plot_from_feat(pos_ext, feat, this_map=mymap):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.scatter(pos_ext[:,0], pos_ext[:,1], c=feat, cmap=this_map, s=400)
    ax.set_xticks([])
    ax.set_yticks([])

    #plt.show()

# Flip every dataset to get a 'new' feature (except for k=0, k=36)
def augment_data(feat_vec, y):
    n_feat = feat_vec.shape[0]

    n_aug = n_feat*2 - 2 # no aug for k=0, k=36

    aug_feat_vec = np.zeros((n_aug, feat_vec.shape[1]))
    aug_feat_vec[:n_feat, :] = feat_vec

    if y.ndim == 1:
        aug_y = np.zeros(n_aug)
    else:
        aug_y = np.zeros((n_aug, y.shape[1]))
    aug_y[:n_feat] = y

    for i in range(n_feat, n_aug):
        orig_feat = feat_vec[i - n_feat]
        orig_y = y[i - n_feat]

        # Rasterize, flip axes, and re-ravel
        new_feat = orig_feat.reshape(6,6)[::-1, ::-1].ravel()

        aug_feat_vec[i] = new_feat
        aug_y[i] = orig_y


    return (aug_feat_vec, aug_y)

# Load in data (energies and methyl positions)
def load_and_prep(fname='sam_pattern_data.dat.npz'):
    ds = np.load(fname)
    energies = ds['energies']
    k_vals = ds['k_vals']

    # (y,z) positions of each of the 36 hexagonal points on the 6x6 grid, flattened
    # shape: (36, 2)
    positions = ds['positions']
    # details the methyl positions of each config
    # Shape: (n_data, 6x6)
    methyl_pos = ds['methyl_pos']
    poly_4 = ds['poly_4']
    poly_5 = ds['poly_5']

    beta_phi_stars = ds['beta_phi_stars']

    n_data = energies.size

    # Total 12x12 hexagonal grid
    pos_ext = gen_pos_grid(8, z_offset=True, shift_y=-1, shift_z=-1)
    pos_ext = positions.copy()

    # patch_idx is list of patch indices in pos_ext 
    #   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
    d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

    tree = cKDTree(pos_ext)
    neighbors = tree.query_ball_tree(tree, r=0.51)

    adj_mat = np.zeros((pos_ext.shape[0], pos_ext.shape[0]), dtype=np.uint8)
    for i in range(pos_ext.shape[0]):
        indices = np.array(neighbors[i])
        adj_mat[i, indices] = 1
        #adj_mat[i,i] = 0

    # shape: (n_data_points, 12*12)
    feat_vec = np.zeros((n_data, pos_ext.shape[0]), dtype=np.float32) # might as well keep this shit small

    for i_dat, methyl_mask in enumerate(methyl_pos):
        feat_vec[i_dat][patch_indices] = methyl_mask

    f_mean = feat_vec.mean()
    f_std = feat_vec.std()


    return feat_vec, energies, poly_4, beta_phi_stars, pos_ext, patch_indices, methyl_pos, adj_mat


def save_net(net, foutname='net.pkl'):
    with open(foutname, 'wb') as fout:
        pickle.dump(net, fout)

def load_net(fname='net.pkl'):
    with open(fname, 'rb') as fin:
        return pickle.load(fin)
