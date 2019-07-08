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


class SAMDataset(data.Dataset):
    base_transform = transforms.Compose([transforms.ToTensor()])
    norm_data = lambda y: (y-y.min())/(y.max()-y.min())

    def __init__(self, feat_vec, y, transform=base_transform, norm_target=False):
        super(SAMDataset, self).__init__()

        if type(feat_vec) is not np.ndarray or type(y) is not np.ndarray:
            raise ValueError("Error: Input feat vec and energies must be type numpy.ndarray") 

        if type(transform) is not torchvision.transforms.Compose:
            raise ValueError("Supplied transform with improper type ({})".format(type(transform)))

        self.ydim = y.ndim
        n_pts = y.shape[0]
        
        self.y = y.astype(np.float32).reshape(n_pts, -1)

        assert feat_vec.ndim == 2
        self.feat_vec = feat_vec.astype(np.float32).reshape(n_pts, 1, -1)

        self.transform = transform
        if norm_target:
            self.y = SAMDataset.norm_data(self.y)

    def __len__(self):
        return len(self.feat_vec)

    def __getitem__(self, index):
        X = self.feat_vec[index]
        y = self.y[index]
        X = self.transform(X).view(1,-1)


        return X, y

    @property
    def n_dim(self):
        return self.feat_vec.shape[-1]

class SAMConvDataset(data.Dataset):
    base_transform = transforms.Compose([transforms.ToTensor()])
    norm_data = lambda y: (y-y.min())/(y.max()-y.min())

    def __init__(self, feat_vec, energies, transform=base_transform, norm_target=False):
        super(SAMConvDataset, self).__init__()

        if type(feat_vec) is not np.ndarray or type(energies) is not np.ndarray:
            raise ValueError("Error: Input feat vec and energies must be type numpy.ndarray") 

        if type(transform) is not torchvision.transforms.Compose:
            raise ValueError("Supplied transform with improper type ({})".format(type(transform)))

        n_pts = energies.shape[0]
        self.energies = energies.astype(np.float32).reshape(-1,1)

        assert feat_vec.ndim == 3
        self.feat_vec = feat_vec.astype(np.float32)

        self.transform = transform
        if norm_target:
            self.energies = SAMDataset.norm_data(self.energies)

    def __len__(self):
        return len(self.feat_vec)

    def __getitem__(self, index):
        X = self.feat_vec[index]
        y = self.energies[index]

        X = torch.tensor(X).view(1, *self.feat_vec.shape[1:])


        return X, y

    @property
    def n_dim(self):
        return self.feat_vec.shape[-1]

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

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

# Simple vanilla NN taking all head group identities as input
class SAMNet(nn.Module):
    def __init__(self, n_hidden=18, n_patch_dim=8):
        super(SAMNet, self).__init__()
        self.fc1 = nn.Linear(n_patch_dim**2, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        self.drop_out = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)

        return out

# two layer net
class SAM2LNet(nn.Module):
    def __init__(self, n_hidden1=9, n_hidden2=18, n_patch_dim=8):
        super(SAM2LNet, self).__init__()
        self.fc1 = nn.Linear(n_patch_dim**2, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, 1)

        self.drop_out = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out
# Sam gcn
class SAMGraphNet(nn.Module):
    def __init__(self, adj_mat, n_hidden1=36, n_hidden2=36, n_out=1):
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

        self.l1 = nn.Linear(n_patch_dim, n_hidden1)
        self.l2 = nn.Linear(n_hidden1, n_hidden2)
        self.o = nn.Linear(n_hidden1, n_out)

        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, x):
        # number of methyl neighbors for each position
        # Shape: (n_data, 64)
        neigh = torch.matmul(x, self.norm_adj)
        out = F.relu(self.l1(neigh))
        out = self.o(out)

        return out

class SAMGraphNet3L(nn.Module):
    def __init__(self, adj_mat, n_hidden=64, n_out=1):
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
        neigh = torch.matmul(x, self.norm_adj)

        out = F.relu(self.l1(neigh))
        #self.drop_out(out)
        neigh = torch.matmul(out, self.norm_adj)
        out = F.relu(self.l2(out))
        self.drop_out(out)
        out = F.relu(self.l3(out))
        #self.drop_out(out)
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

def train(net, criterion, train_loader, test_loader, do_cnn=False, learning_rate=0.01, weight_decay=0, epochs=1000, break_out=None, log_interval=100, loss_fn=None, loss_fn_args=None):

    # create a stochastic gradient descent optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n_dim = train_loader.dataset[0][0].shape[1]
    # Total number of data points
    n_data = len(train_loader.dataset)
    # num training rounds per epoch 
    n_batches = len(train_loader)
    batch_size = train_loader.batch_size

    # total number of training steps
    n_steps = epochs * n_batches
    losses = np.zeros(n_steps)
    # Training loop

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):

            if not do_cnn:
                # resize data from (batch_size, 1, n_input_dim) to (batch_size, n_input_dim)  
                data = data.view(-1, n_dim)
            net_out = net(data)
            
            if loss_fn is None:
                loss = criterion(net_out, target)
            else:
                loss = loss_fn(net_out, target, criterion, *loss_fn_args)
            losses[batch_idx + epoch*n_batches] = loss.item()

            # Back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % log_interval == 0:
                data_test, target_test = iter(test_loader).next()
                if not do_cnn:
                    data_test = data_test.view(-1, n_dim)
                test_out = net(data_test).detach()
                if loss_fn is None:
                    test_loss = criterion(test_out, target_test).item()
                else:
                    test_loss = loss_fn(test_out, target_test, criterion, *loss_fn_args)


                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (valid: {:.6f})'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), test_loss))
                if break_out is not None and test_loss < break_out:
                    return losses

    return losses


