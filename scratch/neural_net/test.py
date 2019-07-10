import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib.util import *
from scratch.neural_net.lib.datautil import partition_data

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


feat_vec, energies, poly, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep()


def test_partition(feat_vec, energies):
    partition = partition_data(feat_vec, energies, n_groups=2)

    X_train1, y_train1, X_test1, y_test1 = next(partition)
    X_train2, y_train2, X_test2, y_test2 = next(partition)

    np.testing.assert_array_equal(X_train1, X_test2)
    np.testing.assert_array_equal(X_train2, X_test1)