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


from IPython import embed
## Try to avoid round-off e


univ = MDAnalysis.Universe('frozen.tpr', 'frozen.gro')
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
#   gridpts npseudo unique points - i.e. all points
#      on an enlarged grid
gridpts = cartesian([x_bounds, y_bounds, z_bounds])
rho_shape = np.ones(gridpts.shape[0])

tree = cKDTree(gridpts)
prot_h_tree = cKDTree(prot_h_atoms.positions)
prot_tree = cKDTree(prot_atoms.positions)

neighbor_list_by_point = prot_h_tree.query_ball_tree(tree, r=3.4)
neighbor_list = itertools.chain(*neighbor_list_by_point)
neighbor_idx = np.unique( np.fromiter(neighbor_list, dtype=int) )

rho_shape[neighbor_idx] = 0
rho = rho_shape.reshape((n_grids[0], n_grids[1], n_grids[2]))


field = RhoField(rho, gridpts)
field.do_DX('field.dx')

pot = np.zeros_like(rho_shape)


prot_atoms.positions -= min_pos
prot_atoms.write('prot_sasa.pdb')
prot_tree = cKDTree(prot_atoms.positions)
mesh = field.meshpts[0]
meshtree = cKDTree(mesh)

res = meshtree.sparse_distance_matrix(prot_tree, max_distance=100)
res = res.toarray()
inv_dist = 1/res


pot = inv_dist.dot(prot_atoms.charges)

n_atoms = mesh.shape[0]
#mesh = mesh[np.newaxis, ...] 


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


