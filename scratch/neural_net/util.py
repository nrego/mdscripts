## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.sam.util import *
from scratch.neural_net.mnist_net import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle

from matplotlib.colors import LinearSegmentedColormap

colors = [(0,0,1), (0,0,0)]
mymap = LinearSegmentedColormap.from_list('mymap', colors, N=100)

do_cnn = False
no_run = False

def plot_pattern(pos_ext, patch_indices, methyl_mask):
    pos = pos_ext[patch_indices]
    plt.plot(pos_ext[:,0], pos_ext[:,1], 'bo')
    plt.plot(pos[methyl_mask, 0], pos[methyl_mask, 1], 'ko')

    plt.show()

def plot_from_feat(pos_ext, feat):
    plt.scatter(pos_ext[:,0], pos_ext[:,1], c=feat, cmap=mymap, s=100)
    plt.show()

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

    n_data = energies.size

    # Total 12x12 hexagonal grid
    pos_ext = gen_pos_grid(8, z_offset=True, shift_y=-1, shift_z=-1)
    #pos_ext = positions.copy()

    # patch_idx is list of patch indices in pos_ext 
    #   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
    d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

    tree = cKDTree(pos_ext)
    neighbors = tree.query_ball_tree(tree, r=0.51)

    adj_mat = np.zeros((pos_ext.shape[0], pos_ext.shape[0]), dtype=np.uint8)
    for i in range(pos_ext.shape[0]):
        indices = np.array(neighbors[i])
        adj_mat[i, indices] = 1
        adj_mat[i,i] = 0

    # shape: (n_data_points, 12*12)
    feat_vec = np.zeros((n_data, pos_ext.shape[0]), dtype=np.float32) # might as well keep this shit small

    for i_dat, methyl_mask in enumerate(methyl_pos):
        feat_vec[i_dat][patch_indices] = methyl_mask

    f_mean = feat_vec.mean()
    f_std = feat_vec.std()
    return feat_vec, energies, poly_4, pos_ext, patch_indices, methyl_pos, adj_mat
    #return ((feat_vec-f_mean)/f_std, energies)

def init_data_and_loaders(X, y, batch_size, norm_target=False, do_cnn=False):
    if do_cnn:
        feat_vec = feat_vec.reshape(-1, 8, 8)
        dataset = SAMConvDataset(X, y, norm_target=norm_target)
    else:
        dataset = SAMDataset(X, y, norm_target=norm_target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

# Split data into N groups - N-1 will be used as training, and remaining as validation
#   In the case of a remainder (likely), the last group will be smaller
def partition_data(X, y, n_groups=1, batch_size=200, do_cnn=do_cnn):
    n_dat = y.shape[0]
    n_cohort = n_dat // n_groups

    # Randomize our data, and therefore our groups
    rand_idx = np.random.permutation(n_dat)
    X_rand = X[rand_idx]
    y_rand = y[rand_idx]

    for k in range(n_groups):
        # slc is indices of validation (excluded from training) data set
        slc = slice(k*n_cohort, (k+1)*n_cohort)

        y_validate = y_rand[slc]
        X_validate = X_rand[slc]
        n_validate = y_validate.shape[0]

        # Get training samples. np.delete makes a copy and **does not** act on array in-place
        if y_rand.ndim == 1:
            y_train = np.delete(y_rand, slc, axis=0)
        elif y_rand.ndim == 2:
            y_train = np.delete(y_rand, slc, axis=0)
        X_train = np.delete(X_rand, slc, axis=0)

        train_loader = init_data_and_loaders(X_train, y_train, batch_size=batch_size, do_cnn=do_cnn)
        test_loader = init_data_and_loaders(X_validate, y_validate, batch_size=n_validate, do_cnn=do_cnn)


        yield (train_loader, test_loader)

def save_net(net, foutname='net.pkl'):
    with open(foutname, 'wb') as fout:
        pickle.dump(net, fout)

def load_net(fname='net.pkl'):
    with open(fname, 'rb') as fin:
        return pickle.load(fin)
