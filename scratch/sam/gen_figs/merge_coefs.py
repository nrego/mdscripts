
import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

import itertools

from sklearn import datasets, linear_model

def unpackbits(x, num_bits):
  xshape = list(x.shape)
  x = x.reshape([-1, 1])
  to_and = 2**np.arange(num_bits).reshape([1, num_bits])

  return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

#get_rank = lambda feat: np.linalg.matrix_rank(np.dot(feat.T, feat))

def get_rank(feat):
    d = feat - feat.mean(axis=0)
    cov = np.dot(d.T, d)

    return np.linalg.matrix_rank(cov)

# Construct all possible states with given size
def build_states(p=2, q=2):
    n = p*q
    n_states = 2**n

    #if n > 8:
    #    print("too many states, exiting")

    states = np.empty(n_states, dtype=object)

    indices = np.arange(n)

    methyl_masks = unpackbits(np.arange(n_states), n).astype(bool)

    for i, methyl_mask in enumerate(methyl_masks):

        states[i] = State(indices[methyl_mask], p, q)

    return states

def load_states(p=2, q=2):
    return np.load('test_p_{:02d}_q_{:02d}.npy'.format(p,q))

def get_feat_vec(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    n_feat = state.N_tot + state.M_int

    # ko, nkoo, and nkcc, so a total of 2*M_tot + 1 coef's
    feat_vec = np.zeros((n_sample, n_feat))


    for i,state in enumerate(states):

        feat_vec[i,:n_edge] = state.edge_oo
        feat_vec[i,n_edge:2*n_edge] = state.edge_cc
        feat_vec[i,2*n_edge:3*n_edge] = state.edge_oc

    return feat_vec

### Merge edge types

p = 3
q = 3
fnames = sorted(glob.glob('test_*npy'))

for fname in fnames:
    fname = 'test_p_{:02d}_q_{:02d}'.format(p,q)
    #states = build_states(p, q)
    p, q = get_p_q(fname)
    states = load_states(p, q)
    feat_vec = get_feat_vec(states)
    n_feat = feat_vec.shape[1]
    rank = get_rank(feat_vec)

    state = states[0]

    print("p: {:02d} q: {:02d}; N_int: {}. N_ext: {}. m_tot: {}. m_int: {}. m_ext: {}. rank: {}".
          format(p,q,state.N_int,state.N_ext,state.n_edges,state.M_int,state.M_ext,rank))


