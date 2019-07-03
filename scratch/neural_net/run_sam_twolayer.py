## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.sam.util import *
from scratch.neural_net.mnist_net import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

do_cnn = False
no_run = False

def plot_pattern(pos_ext, patch_indices, methyl_mask):
    pos = pos_ext[patch_indices]
    plt.plot(pos_ext[:,0], pos_ext[:,1], 'bo')
    plt.plot(pos[methyl_mask, 0], pos[methyl_mask, 1], 'ko')

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

    n_data = energies.size

    # Total 12x12 hexagonal grid
    pos_ext = gen_pos_grid(8, z_offset=True, shift_y=-1, shift_z=-1)

    # patch_idx is list of patch indices in pos_ext 
    #   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
    d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

    tree = cKDTree(pos_ext)
    neighbors = tree.query_ball_tree(tree, r=0.51)

    adj_mat = np.zeros((pos_ext.shape[0], pos_ext.shape[0]), dtype=np.uint8)
    for i in range(pos_ext.shape[0]):
        indices = np.array(neighbors[i])
        adj_mat[i, indices] = 1

    # shape: (n_data_points, 12*12)
    feat_vec = np.zeros((n_data, 64), dtype=np.uint8) # might as well keep this shit small

    for i_dat, methyl_mask in enumerate(methyl_pos):
        feat_vec[i_dat][patch_indices] = methyl_mask

    f_mean = feat_vec.mean()
    f_std = feat_vec.std()
    return feat_vec, energies, pos_ext, patch_indices, methyl_pos, adj_mat
    #return ((feat_vec-f_mean)/f_std, energies)

def init_data_and_loaders(feat_vec, energies, batch_size, norm_target=False, do_cnn=False):
    if do_cnn:
        feat_vec = feat_vec.reshape(-1, 8, 8)
        dataset = SAMConvDataset(feat_vec, energies, norm_target=norm_target)
    else:
        dataset = SAMDataset(feat_vec, energies, norm_target=norm_target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

# Split data into N groups - N-1 will be used as training, and remaining as validation
#   In the case of a remainder (likely), the last group will be smaller
def partition_data(X, y, n_groups=1, batch_size=200, do_cnn=do_cnn):
    n_dat = energies.size
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
        n_validate = y_validate.size

        # Get training samples. np.delete makes a copy and **does not** act on array in-place
        y_train = np.delete(y_rand, slc)
        X_train = np.delete(X_rand, slc, axis=0)

        train_loader = init_data_and_loaders(X_train, y_train, batch_size=batch_size, do_cnn=do_cnn)
        test_loader = init_data_and_loaders(X_validate, y_validate, batch_size=n_validate, do_cnn=do_cnn)


        yield (train_loader, test_loader)

feat_vec, energies, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep()
pos = pos_ext[patch_indices]

data_partition_gen = partition_data(feat_vec, energies, n_groups=7, batch_size=120, do_cnn=do_cnn)
loader = init_data_and_loaders(feat_vec, energies, 200, False, do_cnn)
dataset = loader.dataset
mses = []

for i_round, (train_loader, test_loader) in enumerate(data_partition_gen):

    if no_run:
        break

    if do_cnn:
        net = SAMConvNet(n_channels=12, kernel_size=5, n_hidden=32)
    else:
        net = SAM2LNet(n_hidden1=12, n_hidden2=12)
    # minimize MSE of predicted energies
    criterion = nn.MSELoss()    

    print("Training/validation round {}".format(i_round))
    print("============================")
    print("\n")

    print("...Training")
    losses = train(net, criterion, train_loader, test_loader,do_cnn, learning_rate=0.001, weight_decay=0.0, epochs=3000)
    print("    DONE...")
    print("\n")
    print("...Testing round {}".format(i_round))
    test_X, test_y = iter(test_loader).next()
    # So there are no shenanigans with MSE
    if not do_cnn:
        test_X = test_X.view(-1, 8*8)

    pred = net(test_X).detach()
    mse = criterion(pred, test_y).item()
    print("Validation MSE: {:.2f}".format(mse))

    mses.append(mse)

    if i_round == 0:
        break
    del net, criterion
