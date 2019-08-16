import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib import *
from scratch.interactions.util import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from scipy.spatial import cKDTree

import os, glob

feat_vec, energies, poly, beta_phi_stars, positions, patch_indices, methyl_pos, adj_mat = load_and_prep()
n_dat = feat_vec.shape[0]

aug_feat_vec, aug_y = hex_augment_data(feat_vec, energies)

dataset = SAMConvDataset(aug_feat_vec, aug_y)

idx = 700
out_tensor = np.zeros((6, dataset.X.shape[-2], dataset.X.shape[-1]))
for i in range(6):
    x, y = dataset[idx+i*n_dat]
    out_tensor[i] = x

plot_hextensor(out_tensor[None,...])
plt.show()
