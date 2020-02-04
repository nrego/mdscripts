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

## Torch Datasets for dealing with sam pattern data
#
# Utility functions for constructing data loaders


## Simple SAM dataset
class SAMDataset(data.Dataset):
    
    norm_data = lambda y, y_min, y_max: (y-y_min)/(y_max-y_min)

    def __init__(self, X, y, norm_target=False, y_min=None, y_max=None):
        super(SAMDataset, self).__init__()

        if type(X) is not np.ndarray or type(y) is not np.ndarray:
            raise ValueError("Error: Input feat vec and energies must be type numpy.ndarray") 

        
        if X.shape[0] != y.shape[0]:
            raise ValueError("Error: Different number of input features and targets ({} and {})".format(X.shape[0], y.shape[0]))
        n_pts = X.shape[0]
        
        self.X = torch.from_numpy(X.astype(np.float32).reshape(n_pts, -1))
        self.y = torch.from_numpy(y.astype(np.float32).reshape(n_pts, -1))

        if torch.cuda.is_available():
            print("GPU detected; Initializing SAM dataset with CUDA")
            self.X = self.X.cuda()
            self.test_y = self.y.cuda()

        if norm_target:
            if y_min is None or y_max is None:
                raise ValueError("Must supply minimum and maximum value for y vector if normalizing")
            if self.y_dim > 1:
                assert y_min.size == y_max.size == self.y_dim
                assert ((y_max - y_min) > 0).all()
            else:
                assert y_max > y_min
            self.y = SAMDataset.norm_data(self.y, y_min, y_max)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]

        return X, y

    @property
    def X_dim(self):
        return self.X.shape[-1]

    @property
    def y_dim(self):
        return self.y.shape[-1]

## SAM Conv dataset - transforms X to 1 x 6 x 6 image
class OLDSAMConvDataset(SAMDataset):
    

    def __init__(self, X, y, norm_target=False, y_min=None, y_max=None):
        super(SAMConvDataset, self).__init__(X, y, norm_target, y_min, y_max)

        old_X = self.X.copy()

        self.X = np.zeros((old_X.shape[0], 1, 6, 6), dtype=np.float32)

        for i in range(old_X.shape[0]):
            x = old_X[i]
            x = np.ascontiguousarray(x.reshape(6,6).T[::-1, ::-1])
            self.X[i,0] = x

        self.X = torch.from_numpy(self.X)


class SAMConvDataset(SAMDataset):

    def __init__(self, X, y, norm_target=False, y_min=None, y_max=None, ny=13, nz=13):
        super(SAMConvDataset, self).__init__(X, y, norm_target, y_min, y_max)

        old_X = self.X.detach().numpy().copy()

        self.X = np.zeros((old_X.shape[0], 1, nz, ny), dtype=np.float32)

        for i in range(old_X.shape[0]):
            x = old_X[i]
            x = np.ascontiguousarray(x.reshape(ny, nz).T[::-1,...])
            self.X[i,0] = x

        #self.X = torch.from_numpy(self.X)
        if torch.cuda.is_available():
            self.X = self.X.cuda()

        del old_X

# Split SAM dataset into N random, equally sized groups - N-1 will be used as training, and remaining as validation
#   In the case of a remainder, the remainder is cut off.
def partition_data(X, y, n_groups=1):
    n_dat = y.shape[0]
    # Number of samples in each group
    n_cohort = n_dat // n_groups

    # Randomize our data, and therefore our groups
    rand_idx = np.random.permutation(n_dat)
    X_rand = X[rand_idx]
    y_rand = y[rand_idx]

    for k in range(n_groups):
        # slc is indices of validation (excluded from training) data set
        slc = slice(k*n_cohort, (k+1)*n_cohort)

        # Select the kth group as training
        y_test = y_rand[slc]
        X_test = X_rand[slc]
        assert n_cohort == y_test.shape[0]

        # Get training samples from the n_groups-1 other groups.
        #   np.delete makes a copy and **does not** act on array in-place
        X_train = np.delete(X_rand, slc, axis=0)
        y_train = np.delete(y_rand, slc, axis=0)


        yield (X_train, y_train, X_test, y_test)

