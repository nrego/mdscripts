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

norm = plt.Normalize(-1,1)

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3)

def make_feat(methyl_mask, pos_ext, patch_indices):
    feat = np.zeros(pos_ext.shape[0])
    feat[patch_indices[methyl_mask]] = 1
    feat[patch_indices[~methyl_mask]] = -1

    return feat

def plot_feat(feat, ny=8, nz=8):
    this_feat = feat.reshape(ny, nz).T[::-1, :]

    return this_feat[None, None, ...]

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

    def __init__(self, pt_idx, ny=6, nz=6, shift_y=0, shift_z=0, parent=None, reg=None, e_func=None, mode='build_phob'):
        #embed()
        self.ny = ny
        self.nz = nz
        self.positions = gen_pos_grid(ny, nz, shift_y=shift_y, shift_z=shift_z)
        self.pos_ext = gen_pos_grid(ny+2, nz+2, z_offset=True, shift_y=-1, shift_z=-1)

        _, self.patch_indices = cKDTree(self.pos_ext).query(self.positions, k=1)
        self.non_patch_indices = np.setdiff1d(np.arange(self.pos_ext.shape[0]), self.patch_indices)
        self.nn, self.nn_ext, dd, dd_ext = construct_neighbor_dist_lists(self.positions, self.pos_ext)

        self.N = self.ny*self.nz
        self.ext_count = np.zeros(self.N, dtype=int)
        
        for i in range(self.N):
            self.ext_count[i] = np.intersect1d(self.non_patch_indices, self.nn_ext[i]).size

        self.pt_idx = pt_idx
        self.e_func = e_func
        self.reg = reg

        self.methyl_mask = np.zeros(self.N, dtype=bool)
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
                self._avail_indices = np.delete(np.arange(self.N), self.pt_idx)
            else:
                self._avail_indices = self.pt_idx.copy()

        return self._avail_indices

    @property
    def k_o(self):
        return self.N - self.pt_idx.size

    @property
    def k_c(self):
        return self.pt_idx.size

    @property
    def N_int(self):
        return (self.ny-2)*(self.nz-2)

    @property
    def N_ext(self):
        return 2*(self.ny + self.nz) - 4

    @property
    def n_mm(self):
        n_mm = 0
        for i in self.pt_idx:
            for j in self.nn[i]:
                if j > i and self.methyl_mask[j]:
                    n_mm += 1

        return n_mm

    @property
    def n_mo(self):
        n_mo = 0
        for i in self.pt_idx:
            for j in self.nn[i]:
                n_mo += ~self.methyl_mask[j]

        return n_mo

    @property
    def n_me(self):
        n_me = 0
        for i in self.pt_idx:
            n_me += self.ext_count[i]

        return n_me

    @property
    def n_oo(self):
        n_oo = 0
        for i in np.setdiff1d(np.arange(self.N, dtype=int), self.pt_idx):
            for j in self.nn[i]:
                if j > i and ~self.methyl_mask[j]:
                    n_oo += 1

        return n_oo

    @property
    def n_oe(self):
        n_oe = 0
        for i in np.setdiff1d(np.arange(self.N, dtype=int), self.pt_idx):
            n_oe += self.ext_count[i]

        return n_oe

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
            
            yield State(new_pt_idx, parent=self, reg=self.reg, e_func=self.e_func, mode=self.mode)


    def plot(self, **kwargs):
        #embed()
        feat = make_feat(self.methyl_mask, self.pos_ext, self.patch_indices)
        feat = plot_feat(feat, self.ny+2, self.nz+2)

        new_kwargs = dict()
        if kwargs is not None:

            for k, v in kwargs.items():
                if v is None:
                    continue
                tmp = np.ones(64, dtype=v.dtype)
                tmp[self.patch_indices] = v
                tmp = plot_feat(tmp).T.ravel()
                new_kwargs[k] = tmp

        plot_hextensor(feat, norm=norm, **new_kwargs)

