from __future__ import division

import numpy as np
import MDAnalysis

import argparse

parser = argparse.ArgumentParser('Find electrostatic potential at a collection of points, given a collection of point charges')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='Topology file (containing protein atoms, their positions, and their point charges')
parser.add_argument('-p', '--points', type=str, required=True,
                    help='PDB file containing a collection of points for which to calculate potential')


