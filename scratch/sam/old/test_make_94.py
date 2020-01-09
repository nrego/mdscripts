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

state = State(np.arange(36), 4,9)

univ_oh = MDAnalysis.Universe("whole_oh.gro")

pos_ext = gen_pos_grid(12, z_offset=True, shift_y=-3, shift_z=-3) 
positions = state.positions - pos_ext.min(axis=0)
positions += np.array([0.4330127, -0.75])
pos_ext -= pos_ext.min(axis=0)

#plt.plot(pos_ext[:,0], pos_ext[:,1], 'x')
#plt.plot(positions[:,0], positions[:,1], 'o')
#plt.show()

ag_oh_all = univ_oh.select_atoms("resname OH")
ag_s1_all = ag_oh_all.select_atoms("name S1")
pos_s1_all = ag_s1_all.positions[:,1:]/10
diff = pos_ext.min(axis=0) - pos_s1_all.min(axis=0)
pos_s1_all += diff

#plt.plot(pos_ext[:,0], pos_ext[:,1], 'x')
#plt.plot(pos_s1_all[:,0], pos_s1_all[:,1], 'x')
#plt.plot(positions[:,0], positions[:,1], 'o')

tree = cKDTree(pos_s1_all)
d, patch_idx = tree.query(positions, k=1)
non_patch_idx = np.setdiff1d(np.arange(144), patch_idx)

patch_res = ag_oh_all.residues[patch_idx]
non_patch_res = ag_oh_all.residues[non_patch_idx]
other_res = univ_oh.select_atoms("not resname OH").residues

patch_s1 = patch_res.atoms.select_atoms("name S1")
non_patch_s1 = non_patch_res.atoms.select_atoms("name S1")

u_non_patch = MDAnalysis.core.universe.Merge(non_patch_res.atoms, other_res.atoms)
u_non_patch.atoms.write("non_patch.gro")
patch_res.atoms.write("patch.gro")