import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# One hidden layer net
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()


        self.l1 = nn.Linear(36, 12)
        self.o = nn.Linear(12, 5)

    def forward(self, x):

        out = nn.ReLU(self.l1(x))
        out = self.o(out)

        return out

def loss_poly(net_out, target, criterion, p_min, p_range_mat, xvals):

    # Depends on number of datapoints (batch size), so has to be setup here
    p_min_mat = np.ones((net_out.shape[0], p_min.size), dtype=np.float32)
    p_min_mat *= p_min
    p_min_mat = torch.tensor(p_min_mat)

    pred = torch.matmul(net_out, p_range_mat) + p_min_mat
    pred = torch.matmul(pred, xvals)

    act = torch.matmul(target, p_range_mat) + p_min_mat
    act = torch.matmul(act, xvals)

    loss = criterion(pred, act)

    return loss


feat_vec, energies, poly, pos_ext, patch_indices, methyl_pos, adj_mat = load_and_prep()

p_min = poly.min(axis=0).astype(np.float32)
p_max = poly.max(axis=0).astype(np.float32)
p_range = p_max - p_min
p_range_mat = torch.tensor(np.diag(p_range))
xvals = np.arange(0,124)
xvals = np.array([xvals**4,
                  xvals**3,
                  xvals**2,
                  xvals**1,
                  xvals**0])
xvals = torch.tensor(xvals.astype(np.float32))

net = MyNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_X, train_y, test_X, test_y = next(partition_data(feat_vec, poly, n_groups=5))

train_dataset = SAMDataset(train_X, train_y, norm_target=True, y_min=p_min, y_max=p_max)
test_dataset = SAMDataset(test_X, test_y, norm_target=True, y_min=p_min, y_max=p_max)

del train_X, train_y, test_X, test_y

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

for epoch in range(1000):
    for batch_idx, (train_X, train_y) in enumerate(train_loader):
        







