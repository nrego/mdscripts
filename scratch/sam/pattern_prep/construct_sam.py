
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

## Constructs a SAM surface of given size (ny by nz), using unit_oh
## Unit_oh.gro has proper tilt angle (28.5 deg) in z direction

unit_oh = MDAnalysis.Universe("unit_oh.gro")

## Get waters and wall
conf = MDAnalysis.Universe("confout.gro")

water_and_wall = conf.select_atoms("resname SOL")

## size of square patch
p = 6
q = 36
size = p*q

mapping = []
for i in range(size):
    mapping.append(np.ones(unit_oh.atoms.n_atoms)*i)
mapping = np.concatenate(mapping)

univ = MDAnalysis.Universe.empty(size*unit_oh.atoms.n_atoms, n_residues=size, atom_resindex=mapping, trajectory=True)
univ.add_TopologyAttr('resname')
univ.add_TopologyAttr('names')
univ.add_TopologyAttr('resids')

# In A
pos = 10*gen_pos_grid(p, q, z_offset=True)

for i, this_pos in enumerate(pos):

    res = univ.residues[i]

    # Shift sulfur onto this_pos
    shift = np.array([5, *this_pos]) - unit_oh.atoms[-1].position

    res.atoms.positions = unit_oh.atoms.positions + shift
    res.atoms.names = unit_oh.atoms.names
    res.resname = 'OH'

final_univ = MDAnalysis.core.universe.Merge(univ.atoms, water_and_wall.atoms)
final_univ.atoms.write("blah.gro")

