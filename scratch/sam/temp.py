from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

import matplotlib as mpl

from matplotlib import pyplot as plt

from scipy.spatial import cKDTree
import os

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



mpl.rcParams.update({'axes.labelsize': 60})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':40})

from util import *

homedir = os.environ['HOME']

z_space = 0.5 # 0.5 nm spacing
y_space = np.sqrt(3)/2.0 * z_space
pos = gen_pos_grid(ny=24, z_offset=True)
cent = gen_pos_grid(ny=12, z_offset=True, shift_y=6.0, shift_z=6)

#plt.plot(pos[:,0], pos[:,1], 'x')
#plt.plot(cent[:,0], cent[:,1], 'o')


univ = MDAnalysis.Universe('whole_oh.gro')


sulfurs = univ.select_atoms('name S1')
sulf_pos = sulfurs.positions[:,1:] / 10.0

tree_all = cKDTree(sulf_pos)

pos += sulf_pos.min(axis=0)
cent += sulf_pos.min(axis=0)
tree = cKDTree(cent)
res = tree.query_ball_tree(tree_all, r=0.2)


global_patch_indices = np.unique(res)

patch = univ.residues[global_patch_indices]
patch.atoms.write('patch.gro')

not_patch_indices = np.setdiff1d(np.arange(univ.residues.n_residues), global_patch_indices)
not_patch = univ.residues[not_patch_indices]
not_patch = not_patch.atoms.select_atoms('not (resname SOL and prop x < 18)')
not_patch.write('not_patch.gro')

univ = MDAnalysis.Universe('whole_oh.gro')
univ.add_TopologyAttr('tempfactors')
univ.atoms.tempfactors = 1

oh_patch = univ.residues[-144:]

ch3_patch = MDAnalysis.Universe('ch3_patch.gro').residues[:144]

for oh_res, ch3_res in zip(oh_patch, ch3_patch):
    ag_ref = oh_res.atoms
    oh_res.atoms.tempfactors = 0

    ag_ch3 = ch3_res.atoms

    ch3_shift = ag_ref[-1].position - ag_ch3[-1].position
    ag_ch3.positions += ch3_shift

newuniv = MDAnalysis.core.universe.Merge(univ.atoms[univ.atoms.tempfactors == 1], ch3_patch.atoms)
newuniv.atoms.write('whole_ch3.gro')
univ.atoms.tempfactors = 1

