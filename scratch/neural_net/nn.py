from __future__ import division, print_function

import torch
import torch.nn as nn
import numpy as np
from IPython import embed


class Core(nn.Module):
    def __init__(self, A):
        super(Core, self).__init__()
        # parameters

        # Make sure A - the adjacency matrix - is square and symmetrical
        assert A.shape[0] == A.shape[1]
        assert np.array_equal(A, A.T)
        N = A.shape[0]
        I = np.eye(N)
        A_hat = A + I
        #
        #Normalization for A_hat
        row_sum = np.array(A_hat.sum(axis=1)).squeeze()
        D_inv = np.matrix(np.diag(row_sum**(-0.5)))

        self.N = A.shape[0]
        self.A = torch.tensor(A)
        self.A_hat = torch.tensor(A_hat)
        self.D_inv = torch.tensor(D_inv)

class OLS(Core):
    def __init__(self, A):
        super(OLS, self).__init__(A)
        # parameters

        # Normalized connectivity
        #  and N x N tensor
        pre = torch.matmul(self.D_inv, self.A_hat)
        self.pre = torch.matmul(pre, self.D_inv)
        self.outputSize = 1
        self.hiddenSize = 1
        
        # weights
        self.W1 = torch.randn(1, dtype=torch.float64, requires_grad=True) # N X 1 tensor
        self.inter = torch.randn(1, dtype=torch.float64, requires_grad=True)
        #self.W2 = torch.randn(1, dtype=torch.float64, requires_grad=True) # N X 1 tensor
 

    def forward(self, X):

        self.z2 = torch.matmul(X, self.W1)
        self.o = o = self.z2 + self.inter

        return o
   
class GCN(Core):
    def __init__(self, A):
        super(GCN, self).__init__(A)
        # parameters

        # Normalized connectivity
        #  and N x N tensor
        pre = torch.matmul(self.D_inv, self.A_hat)
        self.pre = torch.matmul(pre, self.D_inv)
        self.outputSize = 1
        self.hiddenSize = A.shape[0]
        
        # weights
        self.W1 = torch.randn(1, dtype=torch.float64, requires_grad=True) # N X 1 tensor
        self.inter = torch.randn(1, dtype=torch.float64, requires_grad=True)
        #self.W2 = torch.randn(1, dtype=torch.float64, requires_grad=True) # N X 1 tensor
 

    def forward(self, X):
        self.conv = torch.matmul(X, self.A).sum(1)
        self.z2 = self.conv * self.W1
        self.h = self.z2.clone()
        self.o = o = self.h + self.inter

        return o

