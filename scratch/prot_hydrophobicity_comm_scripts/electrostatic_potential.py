from __future__ import division

import numpy as np
import MDAnalysis

import argparse

from IPython import embed

parser = argparse.ArgumentParser('Find electrostatic potential at a collection of points, given a collection of point charges')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='Topology file (containing protein atoms, their positions, and their point charges')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Structure file')
parser.add_argument('-d', '--grid', type=float, default=1.0,
                    help='grid resolution, in A (default: 1)')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.top, args.struct)

box = np.ceil(univ.dimensions[:3])

embed()
# Set up marching cube stuff - grids in Angstroms
#  ngrids are grid dimensions of discretized space at resolution ngrids[i] in each dimension
n_grids = (box / args.grid).astype(int)+1

log.info("Initializing grid from initial frame")
log.info("Box: {}".format(box))
log.info("Ngrids: {}".format(n_grids))

# Construct 'gridpts' array, over which we will perform
#    nearest neighbor searching for each heavy prot and water atom
#  NOTE: gridpts is an augmented array - that is, it includes
#  (margin+2) array points in each dimension - this is to reflect pbc
#  conditions - i.e. atoms near the edge of the box can 'see' grid points
#    within a distance (opp edge +- cutoff)
#  I use an index mapping array to (a many-to-one mapping of gridpoint to actual box point)
#     to retrieve appropriate real point indices
x_bounds = np.linspace(0,box[0],n_grids[0])
y_bounds = np.linspace(0,box[1],n_grids[1])
z_bounds = np.linspace(0,box[2],n_grids[2])
