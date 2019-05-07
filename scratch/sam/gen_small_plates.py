from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import networkx as nx
from scipy.spatial import cKDTree

from sklearn import datasets, linear_model

from scipy.integrate import cumtrapz

from util import *

import itertools

from sklearn.cluster import AgglomerativeClustering

from wang_landau import WangLandau

def generate_pattern(univ, univ_ch3, ids_res):
    univ.atoms.tempfactors = 1

    for idx_res in ids_res:
        res = univ.residues[idx_res]
        ag_ref = res.atoms
        res.atoms.tempfactors = 0

        ch3_res = univ_ch3.residues[idx_res]
        ag_ch3 = ch3_res.atoms

        ch3_shift = ag_ref[-1].position - ag_ch3[-1].position
        ag_ch3.positions += ch3_shift

    newuniv = MDAnalysis.core.universe.Merge(univ.atoms[univ.atoms.tempfactors == 1], univ_ch3.residues[ids_res].atoms)

    univ.atoms.tempfactors = 1

    return newuniv

N = 1
univ_oh = MDAnalysis.Universe('whole_oh.gro')
univ_oh.add_TopologyAttr('tempfactors')
univ_ch3 = MDAnalysis.Universe('whole_ch3.gro')
univ_ch3.add_TopologyAttr('tempfactors')

n_tot_res = univ_oh.residues.n_residues
patch_start_idx = n_tot_res - 36

positions = gen_pos_grid(6, z_offset=False)
all_indices = np.arange(36).astype(int)

patch_positions = gen_pos_grid(N)
tree = cKDTree(positions)
d, patch_indices = cKDTree(positions).query(patch_positions, k=1)

new_univ = generate_pattern(univ_oh, univ_ch3, patch_indices+patch_start_idx)
new_univ.atoms.write('whole_ch3_N_{:01d}.gro'.format(N))
