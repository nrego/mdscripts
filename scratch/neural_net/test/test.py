import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib import *


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import logging
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING)

DTYPE = np.float32


feat_vec, energies, poly, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep('data/sam_pattern_data.dat.npz')


# Tests that data partitioning works on 1d feature dataset
def test_partition1D():
    partition = partition_data(feat_vec, energies, n_groups=2)

    X_train1, y_train1, X_test1, y_test1 = next(partition)
    X_train2, y_train2, X_test2, y_test2 = next(partition)

    np.testing.assert_array_equal(X_train1, X_test2)
    np.testing.assert_array_equal(X_train2, X_test1)

def test_partitionND():
    partition = partition_data(feat_vec, poly, n_groups=2)

    X_train1, y_train1, X_test1, y_test1 = next(partition)
    X_train2, y_train2, X_test2, y_test2 = next(partition)

    assert y_train1.ndim == y_train2.ndim == 2

    np.testing.assert_array_equal(X_train1, X_test2)
    np.testing.assert_array_equal(X_train2, X_test1)
    np.testing.assert_array_equal(y_train1, y_test2)

def test_partition_multi1D():
    partition = partition_data(feat_vec, energies, n_groups=5)
    X_train, y_train, X_test, y_test = next(partition)

    expt_cohort_size = feat_vec.shape[0] // 5
     # Training data is everything that's not in testing set
    assert X_train.shape[0] >= expt_cohort_size * 4
    assert X_test.shape[0] == expt_cohort_size

def test_partition_multiND():
    partition = partition_data(feat_vec, poly, n_groups=5)
    X_train, y_train, X_test, y_test = next(partition)

    expt_cohort_size = feat_vec.shape[0] // 5
    assert X_train.shape[0] >= expt_cohort_size * 4
    assert X_test.shape[0] == expt_cohort_size
   
def test_SAMdataset1D():
    dataset = SAMDataset(feat_vec, energies)

    assert dataset.X_dim == 36
    assert dataset.y_dim == 1
    assert len(dataset) == feat_vec.shape[0]

    X, y = dataset[:]
    np.testing.assert_array_almost_equal(feat_vec.astype(np.float32), X)
    np.testing.assert_array_almost_equal(energies.astype(np.float32), y.flatten())
    del dataset

    e_max = energies.max()
    e_min = energies.min()

    norm_energies = (energies - e_min) / (e_max - e_min)

    dataset = SAMDataset(feat_vec, energies, norm_target=True, y_min=e_min, y_max=e_max)

    X, y = dataset[:]
    np.testing.assert_array_almost_equal(feat_vec.astype(np.float32), X)
    np.testing.assert_array_almost_equal(norm_energies.astype(np.float32), y.flatten())

def test_SAMdatasetND():
    dataset = SAMDataset(feat_vec, poly)

    assert dataset.X_dim == 36
    assert dataset.y_dim == poly.shape[1]
    assert len(dataset) == feat_vec.shape[0]

    X, y = dataset[:]
    np.testing.assert_array_almost_equal(feat_vec.astype(np.float32), X)
    np.testing.assert_array_almost_equal(poly.astype(np.float32), y)
    del dataset

    e_max = poly.max(axis=0)
    e_min = poly.min(axis=0)

    norm_poly = (poly - e_min) / (e_max - e_min)

    dataset = SAMDataset(feat_vec, poly, norm_target=True, y_min=e_min, y_max=e_max)

    X, y = dataset[:]
    np.testing.assert_array_almost_equal(feat_vec.astype(np.float32), X)
    np.testing.assert_array_almost_equal(norm_poly.astype(np.float32), y)



