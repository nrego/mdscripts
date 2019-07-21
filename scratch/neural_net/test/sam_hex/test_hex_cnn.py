## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

from scratch.neural_net.lib import *

from model import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import hexagdly

from hexagdly_tools import plot_hextensor

feat_vec, energies, poly, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep('data/sam_pattern_data.dat.npz')

plt.ion()

feat = feat_vec[310]
#plot_from_feat(pos_ext, feat)

# On 6x6 grid and mirrored
myfeat = feat.reshape(6,6).T[::-1, ::-1]
myfeat = torch.tensor(np.ascontiguousarray(myfeat)).reshape(1,1,6,6)

#plot_hextensor(myfeat.reshape(1,1,6,6))

hexconv = hexagdly.Conv2d(1, 1, kernel_size=1, stride=1, bias=False, debug=True)
out = hexconv(myfeat).detach()


dataset = SAMConvDataset(feat_vec, poly)

net = SAMConvNet()

out = net(myfeat)