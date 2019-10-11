from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

import time

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from scratch.sam.util import *

from scratch.neural_net.lib import *

homedir = os.environ["HOME"]
energies, methyl_pos, k_vals, positions, pos_ext, patch_indices, nn, nn_ext, edges, ext_indices, int_indices = extract_data('{}/simulations/pooled_pattern_sample/sam_pattern_data.dat.npz'.format(homedir))
# global indices of non-patch nodes
non_patch_indices = np.setdiff1d(np.arange(pos_ext.shape[0]), patch_indices)

# Locally indexed array: ext_count[i] gives number of non-patch neighbors to patch atom i
ext_count = np.zeros(36, dtype=int)
for i in range(36):
    ext_count[i] = np.intersect1d(non_patch_indices, nn_ext[i]).size
norm = plt.Normalize(-1,1)

def make_feat(methyl_mask):
    feat = np.zeros(pos_ext.shape[0])
    feat[patch_indices[methyl_mask]] = 1
    feat[patch_indices[~methyl_mask]] = -1

    return feat

def plot_feat(feat, ny=8, nz=8):
    this_feat = feat.reshape(ny, nz).T[::-1, :]

    return this_feat.reshape(1,1,ny,nz)

def get_energy(pt_idx, m_mask, nn, ext_count, reg):
    coef1, coef2, coef3 = reg.coef_
    inter = reg.intercept_
    mm = 0
    mo_int = 0
    mo_ext = 0
    for m_idx in pt_idx:
        for n_idx in nn[m_idx]:
            if n_idx > m_idx:
                mm += m_mask[n_idx]
            mo_int += ~m_mask[n_idx]
        mo_ext += ext_count[m_idx]
    
    return inter + coef1*mm + coef2*mo_int + coef3*mo_ext

class State:
    positions = positions.copy()
    pos_ext = pos_ext.copy()
    patch_indices = patch_indices.copy()
    ext_count = ext_count.copy()
    nn = nn.copy()

    def __init__(self, pt_idx, parent=None, reg=None, e_func=None, mode='build_phob'):
        self.pt_idx = pt_idx
        self.nn = nn
        self.ext_count = ext_count
        self.e_func = e_func
        self.reg = reg

        self.methyl_mask = np.zeros(36, dtype=bool)
        self.methyl_mask[self.pt_idx] = True

        self.parent = parent
        self.children = list()

        self._energy = None
        self._avail_indices = None

        self.mode = mode

    @property
    def avail_indices(self):
        if self._avail_indices is None:
            if self.mode == 'build_phob':
                self._avail_indices = np.delete(np.arange(36), self.pt_idx)
            else:
                self._avail_indices = self.pt_idx.copy()

        return self._avail_indices

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    @property
    def energy(self):
        if self._energy is None:
            self._energy = self.e_func(self.pt_idx, self.methyl_mask, self.nn, self.ext_count, self.reg)

        return self._energy

    def gen_next_pattern(self):

        for idx in self.avail_indices:
            if self.mode == 'build_phob':
                new_pt_idx = np.append(self.pt_idx.copy(), idx).astype(int)
            else:
                idx = np.where(self.pt_idx == idx)[0].item()
                new_pt_idx = np.delete(self.pt_idx, idx).astype(int)
            
            yield State(new_pt_idx, self, self.reg, self.e_func, mode=self.mode)


    def plot(self):
        feat = make_feat(self.methyl_mask)
        feat = plot_feat(feat)

        plot_hextensor(feat, norm=norm)

