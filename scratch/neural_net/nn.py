from __future__ import division, print_function

import torch
import torch.nn as nn
import numpy as np
from IPython import embed

class GCN(nn.Module):
    def __init__(self, A):
        super(GCN, self).__init__()
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
        self.A_hat = torch.tensor(A_hat)
        self.D_inv = torch.tensor(D_inv)
        # Normalized connectivity
        #  and N x N tensor
        pre = torch.matmul(self.D_inv, self.A_hat)
        self.pre = torch.matmul(pre, self.D_inv)
        self.outputSize = 1
        self.hiddenSize = 4
        
        # weights
        self.W1 = torch.randn(self.N, self.hiddenSize, dtype=torch.float64, requires_grad=True) # N X 4 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize, dtype=torch.float64, requires_grad=True) # 3 X 2 tensor
        
    def forward(self, X):
        self.conv1 = torch.matmul(self.pre, X) # N x D, where D is number of att for each node
        self.z1 = torch.matmul(self.conv1, self.W1)

        self.conv2 = torch.matmul(self.pre, self.z1)
        self.z2 = torch.matmul(self.conv2, self.W2)
        self.o = o = self.z2.clone()
        return o
    
    def backward(self, X, y, o):
        self.loss = (o - y).pow(2).sum()

        return self.loss.backward()
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))