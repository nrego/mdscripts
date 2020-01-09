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
from scratch.sam.util import *
import itertools
from sklearn.cluster import AgglomerativeClustering
from wang_landau import WangLandau


N = 5
univ_oh = MDAnalysis.Universe('whole_oh.gro')
univ_oh.add_TopologyAttr('tempfactors')
univ_ch3 = MDAnalysis.Universe('whole_ch3.gro')
univ_ch3.add_TopologyAttr('tempfactors')

n_tot_res = univ_ch3.residues.n_residues
patch_start_idx = n_tot_res - N**2

# N x N patch atoms
patch_ch3 = univ_ch3.select_atoms("resname CH3").select_atoms("name S")
patch_oh = univ_oh.residues[-36:].atoms.select_atoms("name S1")
pre_patch_oh = univ_oh.residues[:-36].atoms.select_atoms("name S1")
non_oh = univ_oh.select_atoms("not resname OH")

tree = cKDTree(patch_oh.positions)
# patch_indices shape (N*N), gives indices of patch_oh that are on the actual patch
d, patch_indices = tree.query(patch_ch3.positions, k=1)
# non_patch_indices contains all OH that should be moved
non_patch_indices = np.setdiff1d(np.arange(36), patch_indices)
new_univ = MDAnalysis.core.universe.Merge(pre_patch_oh.residues.atoms, patch_oh[non_patch_indices].residues.atoms, non_oh, patch_oh[patch_indices].residues.atoms)

new_univ.atoms.write("whole_oh.gro")
