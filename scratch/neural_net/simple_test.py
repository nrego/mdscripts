## TEST NUMBER OF EDGE TYPES (MODEL 2) AGAINST NN ##

import numpy as np

from scratch.sam.util import *
from scratch.neural_net.mnist_net import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def load_stuff(filedump='sam_pattern_data.dat.npz', k_eff_dump='k_eff_all.dat.npy'):
    ds = np.load(filedump)

    energies = ds['energies']
    k_vals = ds['k_vals']
    
    methyl_pos = ds['methyl_pos']
    positions = ds['positions']


    k_eff_all_shape = np.load(k_eff_dump)

    payload = (energies, k_vals, methyl_pos, positions, k_eff_all_shape)


    return payload

def init_data_and_loaders(feat_vec, energies, batch_size=221):
    
    dataset = SAMDataset(feat_vec, energies, norm_target=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

## k_eff_all_shape is an exhaustive list of the connection type of every
# edge for every pattern
## n_mm n_oo  n_mo ##
# Conn types:   mm  oo  mo
# shape: (n_samples, n_edges, n_conn_type)
energies, k_vals, methyl_pos, positions, k_eff_all_shape = load_stuff()

k_vals_both = np.hstack((k_vals[:,None], 36-k_vals[:,None]))

n_edges = k_eff_all_shape.shape[0]
# n_mm, n_oo, n_mo
k_eff_one_edge = k_eff_all_shape.sum(axis=1)

# k_c, n_mm
feat_vec = np.dstack((k_vals, k_eff_one_edge[:,0])).squeeze(axis=0)
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies, do_ridge=False)
print("average mse, linear fit: {:0.4f}".format(perf_mse.mean()))



### Train ###
net = TestM2()
# minimize MSE of predicted energies
criterion = nn.MSELoss()
loader = init_data_and_loaders(feat_vec, energies)
dataset = loader.dataset

data, target = iter(loader).next()

losses = train(net, criterion, loader)
