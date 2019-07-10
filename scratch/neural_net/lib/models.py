import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from IPython import embed




# Runs basic linear regression on our number of edges
#   For testing 
class TestSAMNet(nn.Module):
    def __init__(self, n_dim=2):
        super(TestSAMNet, self).__init__()
        self.fc1 = nn.Linear(n_dim, 1)

    def forward(self, x):
        x = self.fc1(x)

        return x

    @property
    def coef_(self):
        return self.fc1.weight.detach().numpy()

    @property
    def intercept_(self):
        return self.fc1.bias.item()

# One hidden layer net
class SAMNet1L(nn.Module):
    def __init__(self, n_hidden=36, n_out=1):
        super(SAMGraphNet, self).__init__()

        n_patch_dim = adj_mat.shape[0]
        self.adj_mat = torch.tensor(adj_mat.astype(np.float32))
        node_deg = np.diag(adj_mat.sum(axis=0)).astype(np.float32)
        mask = node_deg > 0
        d_inv = node_deg.copy()
        d_inv[mask] = node_deg[mask]**-1
        self.node_deg = torch.tensor(node_deg)
        self.d_inv = torch.tensor(d_inv)

        # Normalized adj mat
        self.norm_adj = torch.matmul(self.adj_mat, self.d_inv)
        self.l1 = nn.Linear(n_patch_dim, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.o = nn.Linear(n_hidden, n_out)

        self.weight = torch.rand(36)

        self.drop_out = nn.Dropout(p=0.5)

    def forward(self, x):
        # number of methyl neighbors for each position
        # Shape: (n_data, 36)

        #neigh = torch.matmul(x, self.norm_adj)
        out = F.relu(self.l1(x))
        self.drop_out(out)
        #out = F.relu(self.l2(out))
        #self.drop_out(out)
        out = self.o(out)

        return out

class SAMGraphNet3L(nn.Module):
    def __init__(self, adj_mat, n_hidden=36, n_out=1):
        super(SAMGraphNet3L, self).__init__()

        n_patch_dim = adj_mat.shape[0]
        self.adj_mat = torch.tensor(adj_mat.astype(np.float32))
        node_deg = np.diag(adj_mat.sum(axis=0)).astype(np.float32)
        mask = node_deg > 0
        d_inv = node_deg.copy()
        d_inv[mask] = node_deg[mask]**-1
        self.node_deg = torch.tensor(node_deg)
        self.d_inv = torch.tensor(d_inv)

        # Normalized adj mat
        self.norm_adj = torch.matmul(self.adj_mat, self.d_inv)

        self.l1 = nn.Linear(n_patch_dim, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, n_hidden)
        self.o = nn.Linear(n_hidden, n_out)

        self.drop_out = nn.Dropout(p=0.5)

    def forward(self, x):
        # number of methyl neighbors for each position
        # Shape: (n_data, 64)

        out = F.relu(self.l1(x))
        self.drop_out(out)
        out = F.relu(self.l2(out))
        self.drop_out(out)
        out = F.relu(self.l3(out))
        self.drop_out(out)
        out = self.o(out)

        return out

# Sam cnn
class SAMConvNet(nn.Module):
    def __init__(self, n_channels=4, kernel_size=3, n_patch_dim=8, n_hidden=18):
        super(SAMConvNet, self).__init__()
        l1_outdim = (n_patch_dim - kernel_size) + 1
        l1_outdim_pool = (l1_outdim - 2)/2 + 1
        assert l1_outdim == int(l1_outdim)
        assert l1_outdim_pool == int(l1_outdim_pool)
        self.l1_outdim = int(l1_outdim)
        self.l1_outdim_pool = int(l1_outdim_pool)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(self.l1_outdim_pool * self.l1_outdim_pool * n_channels, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

