from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed
from mdtools import MDSystem
import cPickle as pickle
import argparse

parser = argparse.ArgumentParser('Find buried atoms, surface atoms (and mask), and dewetted atoms')
parser.add_argument('-s', '--topology', type=str, required=True,
                    help='Input topology file')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Input structure file')
parser.add_argument('--ref', type=str, required=True,
                    help='rho_data_dump of the reference structure')
parser.add_argument('--rhodata', type=str, required=True, 
                    help='rho_data_dump file for which to find dewetted atoms')
parser.add_argument('-nb', default=5, type=float,
                    help='Solvent exposure criterion for determining buried atoms from reference')

args = parser.parse_args()

sys = MDSystem(args.topology, args.struct)

embed()