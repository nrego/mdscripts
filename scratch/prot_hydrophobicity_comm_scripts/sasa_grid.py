#Instantaneous interface
# nrego sept 2015

from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
from argparse import ArgumentTypeError
import logging

import cPickle as pickle

import MDAnalysis
from MDAnalysis import SelectionError
#from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

from scipy.spatial import cKDTree
import itertools
#from skimage import measure
import mdtraj as md

from rhoutils import rho, cartesian
from mdtools import ParallelTool, Subcommand

from fieldwriter import RhoField

from constants import k

from IPython import embed
## Try to avoid round-off e

parser = argparse.ArgumentParser('Find electrostatic potential at a collection of points, given a collection of point charges')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='Topology file (containing protein atoms, their positions, and their point charges)')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Structure file (GRO or PDB)')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.top, args.struct)
prot_atoms = univ.select_atoms('protein')
prot_h_atoms = univ.select_atoms('protein and not name H*')

# draw bounding box around the protein w/ 0.5 A buffer
min_pos = np.floor(prot_atoms.positions.min(axis=0)) - 5
max_pos = np.ceil(prot_atoms.positions.max(axis=0)) + 5

# setup grid points
n_grids = (max_pos - min_pos).astype(int) + 1

x_bounds = np.linspace(min_pos[0], max_pos[0], n_grids[0])
y_bounds = np.linspace(min_pos[1], max_pos[1], n_grids[1])
z_bounds = np.linspace(min_pos[2], max_pos[2], n_grids[2])

# gridpts array shape: (n_pts, 3)
# 
# set up grid around protein's bounding box
gridpts = cartesian([x_bounds, y_bounds, z_bounds])
rho_shape = np.ones(gridpts.shape[0])

tree = cKDTree(gridpts)
prot_tree = cKDTree(prot_atoms.positions)
prot_h_tree = cKDTree(prot_h_atoms.positions)

# Get all gridpoints within 3.4 A of protein (i.e. within its SASA)
neighbor_list_by_point = prot_h_tree.query_ball_tree(tree, r=3.4)
neighbor_list = itertools.chain(*neighbor_list_by_point)
neighbor_idx = np.unique( np.fromiter(neighbor_list, dtype=int) )

rho_shape[neighbor_idx] = 0
rho = rho_shape.reshape((n_grids[0], n_grids[1], n_grids[2]))

# Output SASA density map
field = RhoField(rho, gridpts)
field.do_DX('field.dx')



# Output protein (translated)
prot_atoms.positions -= min_pos
prot_atoms.write('prot_sasa.pdb')


# Calculate electrostatic potential at each SASA isosurface vert
pot = np.zeros_like(rho_shape)
mesh = field.meshpts[0]
meshtree = cKDTree(mesh)

print('Calculating distance matrix')
prot_tree = cKDTree(prot_atoms.positions)
res = meshtree.sparse_distance_matrix(prot_tree, max_distance=1000)

res = res.power(-1)

beta = 1 / (k*300)
# In units of kT / e^2
ke = beta * 138.935485
charges = prot_atoms.charges
net_charge = charges.sum()
print("net charge: {:f}".format(net_charge))
offset_charge = -net_charge / prot_atoms.n_atoms
charges += offset_charge
#embed()
pot = res.dot(charges) * ke

n_atoms = mesh.shape[0]

pot = np.clip(pot, -9.9, 99)
#embed()
# Output SASA colored by electrostatic potential
top = md.Topology()
c = top.add_chain()

cnt = 0
for i in range(n_atoms):
    cnt += 1
    r = top.add_residue('II', c)
    a = top.add_atom('II', md.element.get_by_symbol('VS'), r, i)

with md.formats.PDBTrajectoryFile('field.pdb', 'w') as f:
    # Mesh pts have to be in nm
    f.write(mesh, top, bfactors=pot)


