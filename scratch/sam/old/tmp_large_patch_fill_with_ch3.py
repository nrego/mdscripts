from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

import matplotlib as mpl

from matplotlib import pyplot as plt

from scipy.spatial import cKDTree
import os, glob, pathlib
from scratch.neural_net import *
from scratch.sam.util import *

## Given two SAMs of same size (all oh, all ch3)
##   Fill a given patch in the SAM with all CH3

def generate_pattern(univ, univ_ch3, ids_res):
    univ.atoms.tempfactors = 1
    #embed()
    if ids_res.size == 0:
        return univ
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


patch_size = 3
patch_size_2 = 12
N = patch_size*patch_size_2
pos_idx = np.arange(N, dtype=int)

state = State(pos_idx, patch_size, patch_size_2)

univ_oh = MDAnalysis.Universe('../whole_oh.gro')
univ_ch3 = MDAnalysis.Universe('../whole_ch3.gro')
univ_oh.add_TopologyAttr('tempfactors')
univ_ch3.add_TopologyAttr('tempfactors')

assert univ_oh.select_atoms("resname OH").residues.n_residues == univ_ch3.residues.n_residues


oh_s = univ_oh.select_atoms('name S1')
# Positions of entire grid
first_oh_pos = np.round(oh_s.positions[0,1:], 2)
#pos_ext = gen_pos_grid(12, z_offset=True) * 10
pos_ext = gen_pos_grid(6, 36, z_offset=True) * 10
pos_ext += first_oh_pos - pos_ext[0]
y_pos = np.unique(pos_ext[:,0])
z_pos = np.unique(pos_ext[:,1])

start_pos = np.array([y_pos[2], z_pos[3]])
# Patch position
positions = state.positions * 10
shift = start_pos - positions[0]
positions += shift


plt.plot(pos_ext[:,0], pos_ext[:,1], 'x')
plt.plot(positions[:,0], positions[:,1], 'o')
plt.show()

tree = cKDTree(pos_ext)
# Gives indices of ch3 residues and oh resdiues that we'll swap
d, indices = tree.query(positions, k=1)
assert np.unique(indices).size == indices.size == N

new_univ = generate_pattern(univ_oh, univ_ch3, indices)

new_univ.atoms.write('{}_{}.gro'.format(patch_size, patch_size_2))

