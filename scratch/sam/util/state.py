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

def plot_feat(feat, p=8, q=8):
    this_feat = feat.reshape(p, q).T[::-1, :]

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

    def __init__(self, pt_idx, p=6, q=6, shift_y=0, shift_z=0, parent=None, reg=None, e_func=None, mode='build_phob'):
        #embed()
        self.p = p
        self.q = q
        self.positions = gen_pos_grid(p, q, shift_y=shift_y, shift_z=shift_z)
        self.pos_ext = gen_pos_grid(p+2, q+2, z_offset=True, shift_y=-1, shift_z=-1)

        # Patch_indices is a mapping of local (patch) index to (global) patch index
        #   patch_index[local_i] = global_i
        _, self.patch_indices = cKDTree(self.pos_ext).query(self.positions, k=1)
        self.non_patch_indices = np.setdiff1d(np.arange(self.pos_ext.shape[0]), self.patch_indices)
        self.nn, self.nn_ext, dd, dd_ext = construct_neighbor_dist_lists(self.positions, self.pos_ext)

        self.N = self.p*self.q
        # List, keyed by *local* indices, of all external edges made by local patch idx i
        self.ext_count = np.zeros(self.N, dtype=int)
        # List, keyed by *local* indices, of all internal edges made by local patch idx i 
        #.  Note: if we sum this up, this will be 2*M_int, since we're double counting internal edges
        self.int_count = np.zeros_like(self.ext_count)

        # Shape (N, N): adj_mat[i,j] = 1 if i,j nearest neighbors, 0 otherwise
        #   Note: symmetric matrix with 0's on the diagonal
        self.adj_mat = np.zeros((self.N, self.N), dtype=int)
        
        for i in range(self.N):
            self.ext_count[i] = np.intersect1d(self.non_patch_indices, self.nn_ext[i]).size
            self.int_count[i] = self.nn[i].size
            self.adj_mat[i][self.nn[i]] = 1
            # Sanity; each node must make 6 edges
            assert self.ext_count[i] + self.int_count[i] == 6

        assert self.adj_mat.diagonal().max() == 0
        assert np.array_equal(self.adj_mat, self.adj_mat.T)

        self.pt_idx = np.atleast_1d(pt_idx)
        self.e_func = e_func
        self.reg = reg

        self.methyl_mask = np.zeros(self.N, dtype=bool)
        self.methyl_mask[self.pt_idx] = True

        # Get all edges made by patch atoms (global indices), avoid double counting edges
        self.edges, self.edges_ext_indices, self.edges_periph_indices = enumerate_edges(self.positions, self.nn_ext, self.patch_indices, self.nodes_peripheral)
        self.edges_int_indices = np.setdiff1d(np.arange(self.n_edges), self.edges_ext_indices)
        self.edges_buried_indices = np.setdiff1d(self.edges_int_indices, self.edges_periph_indices)

        # Now identify what each edge is
        self.edge_oo, self.edge_cc, self.edge_oc = construct_edge_feature(self.edges, self.edges_ext_indices, self.patch_indices, self.methyl_mask)
        assert self.edge_cc.sum() == self.n_cc
        assert self.edge_oo.sum() == self.n_oo + self.n_oe
        assert self.edge_oc.sum() == self.n_oc + self.n_ce

        assert (self.edge_cc + self.edge_oo + self.edge_oc).sum() == self.n_edges
        
        assert self.M_ext == self.edges_ext_indices.size
        assert self.M_ext + self.M_int == self.n_edges == self.edges.shape[0]

        ## number of internal edges

        self.parent = parent
        self.children = list()

        self._energy = None
        self._avail_indices = None

        self.mode = mode


        # List of each (local indexed) node to the edges it makes
        nodes_to_int_edges = []
        nodes_to_ext_edges = []
        edge_indices = np.arange(self.n_edges)
        # Mask of all external edges
        edge_ext_mask = np.ones(self.n_edges, dtype=bool)
        edge_ext_mask[self.edges_int_indices] = False
        for i_local, i_global in enumerate(self.patch_indices):
            # Mask of all edges made by node i
            edge_mask = self.edges[:,0] == i_global

            # Indices of edges made by this node (external and internal edges)
            this_ext_edge_indices = edge_indices[(edge_mask & edge_ext_mask)]
            this_int_edge_indices = edge_indices[(edge_mask & ~edge_ext_mask)]

            nodes_to_int_edges.append(this_int_edge_indices)
            nodes_to_ext_edges.append(this_ext_edge_indices)


        # List of each node's external and internal edges
        #.  key: node's (local) index
        #.  value: indices of edges (external or internal) made by node i
        self.nodes_to_int_edges = np.array(nodes_to_int_edges)
        self.nodes_to_ext_edges = np.array(nodes_to_ext_edges)

    @property
    def avail_indices(self):
        if self._avail_indices is None:
            if self.mode == 'build_phob':
                self._avail_indices = np.delete(np.arange(self.N), self.pt_idx)
            else:
                self._avail_indices = self.pt_idx.copy()

        return self._avail_indices

    # Local indices of peripheral patch atoms
    @property
    def ext_indices(self):
        return np.arange(self.N, dtype=int)[self.ext_count>0]
    
    # Local indices of internal patch atoms
    @property
    def int_indices(self):
        return np.arange(self.N, dtype=int)[self.ext_count==0]

    @property
    def P(self):
        return self.p
    @property
    def Q(self):
        return self.q

    @property
    def N_tot(self):
        return self.p*self.q

    @property
    def k_o(self):
        return self.N - self.pt_idx.size

    @property
    def k_c(self):
        return self.pt_idx.size

    @property
    def N_int(self):
        return self.p*self.q - 2*(self.p + self.q) + 4

    @property
    def N_ext(self):
        return 2*(self.p + self.q) - 4

    @property
    def M_int(self):
        return 3*self.N_int + 2*self.N_ext - 3

    @property
    def M_ext(self):
        return 2*self.N_ext + 6

    @property
    def n_cc(self):
        n_cc = 0
        for i in self.pt_idx:
            for j in self.nn[i]:
                if j > i and self.methyl_mask[j]:
                    n_cc += 1

        return n_cc

    @property
    def n_oc(self):
        n_oc = 0
        for i in self.pt_idx:
            for j in self.nn[i]:
                n_oc += ~self.methyl_mask[j]

        return n_oc

    @property
    def n_ce(self):
        n_ce = 0
        for i in self.pt_idx:
            n_ce += self.ext_count[i]

        return n_ce

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

    @property
    def n_edges(self):
        return self.edges.shape[0]

    # Returns the (local) mask of all buried nodes
    @property
    def nodes_buried(self):
        return self.ext_count == 0

    @property
    def nodes_peripheral(self):
        return self.ext_count > 0

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
            
            yield State(new_pt_idx, parent=self, p=self.p, q=self.q, mode=self.mode)


    def plot(self, noedge=False, do_annotate=False, **kwargs):

        feat = make_feat(self.methyl_mask, self.pos_ext, self.patch_indices)
        feat = plot_feat(feat, self.p+2, self.q+2)
        if noedge:
            feat[feat==0] = -1

        #new_kwargs = dict()
        #if kwargs is not None:

        #    for k, v in kwargs.items():
        #        if v is None:
        #            continue
        #        tmp = np.ones(64, dtype=v.dtype)
        #        tmp[self.patch_indices] = v
        #        tmp = plot_feat(tmp).T.ravel()
        #        new_kwargs[k] = tmp
        new_kwargs = kwargs
        
        plot_hextensor(feat, norm=norm, **new_kwargs)

        if do_annotate:
            pos = 2*self.positions
            max_y = 2*self.pos_ext[:,1].max()
            pos += np.array([np.sqrt(3)/2, -max_y])

            ax = plt.gca()

            for i, (x,y) in enumerate(pos):
                ax.annotate(i, xy=(x-0.025,y-0.025), fontsize='xx-large', color='darkorange')


    def plot_edges(self, do_annotate=False, annotation=None, colors=None, line_styles=None, line_widths=None, ax=None, symbols=None):
        if ax is None:
            ax = plt.gca()

        if annotation is None:
            annotation = np.arange(self.edges.shape[0])

        this_pos_ext = 2*self.pos_ext
        #max_x = this_pos_ext[:,0].max()
        max_y = this_pos_ext[:,1].max()
        this_pos_ext += np.array([np.sqrt(3)/2, -max_y])

        for i_edge, (global_i, global_j) in enumerate(self.edges):

            assert global_i in self.patch_indices
            local_i = np.where(self.patch_indices==global_i)[0].item()

            # Is global point j within the patch??
            j_int = global_j in self.patch_indices


            if symbols is None:
                i_symbol = 'ko'
            else:
                i_symbol = symbols[local_i]

            if line_styles is None:
                edge_style = '-' 
            else:
                edge_style = line_styles[i_edge]

            if line_widths is None:
                line_width = 3
            else:
                line_width = line_widths[i_edge]

            ax.plot(this_pos_ext[global_i,0], this_pos_ext[global_i,1], i_symbol, markersize=12, zorder=3)
            if not j_int:
                ax.plot(this_pos_ext[global_j,0], this_pos_ext[global_j,1], 'rx', markersize=12, zorder=3)

            if colors is not None:
                this_color = colors[i_edge]
            else:
                this_color = 'k'
            ax.plot([this_pos_ext[global_i,0], this_pos_ext[global_j,0]], [this_pos_ext[global_i,1], this_pos_ext[global_j,1]], color=this_color, linestyle=edge_style, linewidth=line_width)

            midpt = (this_pos_ext[global_i] + this_pos_ext[global_j]) / 2.0

            if do_annotate:
                ax.annotate(annotation[i_edge], xy=midpt-0.025, fontsize='xx-large', color='darkorange')


