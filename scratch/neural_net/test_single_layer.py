import networkx as nx
from networkx import karate_club_graph, to_numpy_matrix
from scratch.neural_net import *
import torch

from util import *

from scipy.spatial import cKDTree

def gen_X(methyl_masks, patch_indices, N):
    X = np.zeros((methyl_masks.shape[0], N)).astype(np.float64)
    for i in range(methyl_masks.shape[0]):
        X[i, patch_indices[methyl_masks[i]]] = 1
    #X = torch.tensor(X)
    return X

ds = np.load('sam_pattern_data.dat.npz')
positions = ds['positions']
energies = ds['energies']
methyl_pos = ds['methyl_pos']

n_dat = energies.shape[0]

pos_grid = gen_pos_grid(ny=6, nz=None, z_offset=False, shift_y=0, shift_z=0)

N = pos_grid.shape[0]

tree = cKDTree(pos_grid)
center_patch_tree = cKDTree(positions)

# Patch indices contains the global indices of each patch position
d, patch_indices = tree.query(positions, k=1)

## Generate adjacency matrix
res = tree.query_ball_tree(tree, r=0.501)

A = np.zeros((N, N))
for i in range(N):
    i_neigh = np.array(res[i])

    A[i,i_neigh] = 1

assert np.array_equal(A, A.T)

net = GCN(A)
with torch.no_grad():
    net.inter[0] = 220.0

learning_rate = 1e-14

n_iter = 500000
losses = np.zeros(n_iter)
dat_idx = 0
batch_size = 1
rand = np.random.choice(n_dat, n_dat, replace=False)
for i in range(n_iter):
    if i % 1000 == 0:
        print("iter: {}".format(i))
        print("mse: {:0.2f}".format(losses[i-1] ))
        #print("w1: {:0.2f}".format(net.W1.item()))
        print("inter: {:0.2f}\n".format(net.inter.item()))
    if dat_idx >= n_dat:
        #rand = np.random.choice(n_dat, n_dat, replace=True)
        dat_idx = 0

    methyl_masks = methyl_pos[rand][dat_idx:dat_idx+batch_size]
    y = torch.tensor(energies[rand][dat_idx:dat_idx+batch_size])

    X = torch.tensor(gen_X(methyl_masks, patch_indices, N))
    y_pred = net.forward(X)
    loss = (y_pred - y).pow(2).sum() / batch_size
 
    loss.backward()

    with torch.no_grad():
        
        net.W1 -= learning_rate*net.W1.grad
        if i%1000 == 0:
            print("  w1_grad: {:0.2f}".format(net.W1.grad.item()))
            print("  inter_grad: {:0.2f}".format(net.inter.grad.item()))
        net.W1.grad.zero_()

        net.inter -= learning_rate*net.inter.grad
        net.inter.grad.zero_()

    losses[i] = loss.item()

    dat_idx += 1
