import numpy as np
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

    def __init__(self, feat_vec, energies, transform=base_transform, norm_target=True):
        super(SAMDataset, self).__init__()

        if type(feat_vec) is not np.ndarray or type(energies) is not np.ndarray:
            raise ValueError("Error: Input feat vec and energies must be type numpy.ndarray") 

        if type(transform) is not torchvision.transforms.Compose:
            raise ValueError("Supplied transform with improper type ({})".format(type(transform)))

        self.energies = energies.astype(np.float32)
        self.feat_vec = feat_vec.astype(np.float32).reshape(-1,1,2)

        self.transform = transform
        if norm_target:
            self.energies = SAMDataset.norm_data(self.energies)

    def __len__(self):
        return len(self.feat_vec)

    def __getitem__(self, index):
        X = self.feat_vec[index]
        y = self.energies[index]

        X = self.transform(X).view(1,-1)


        return X, y

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x)

class TestM2(nn.Module):
    def __init__(self):
        super(TestM2, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        return x



def train(net, criterion, train_loader, learning_rate=0.001, epochs=1000, log_interval=10):

    # create a stochastic gradient descent optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0)

    n_dim = train_loader.dataset[0][0].shape[1]
    # Total number of data points
    n_data = len(train_loader.dataset)
    # num training rounds per epoch (number data points / batch size)
    n_rounds = len(train_loader)
    batch_size = int(n_data / n_rounds)
    if n_data % n_rounds:
        batch_size += 1

    # total number of training steps
    n_steps = epochs * n_rounds
    losses = np.zeros(n_steps)
    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, n_input_dim) to (batch_size, n_input_dim)
            
            data = data.view(-1, n_dim)
            #embed()
            optimizer.zero_grad()
            net_out = net(data)
            
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if epoch % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    return losses


