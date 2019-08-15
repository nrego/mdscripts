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

aug_feat_vec, aug_y = hex_augment_data(feat_vec, energies)

dataset = SAMConvDataset(aug_feat_vec, aug_y)