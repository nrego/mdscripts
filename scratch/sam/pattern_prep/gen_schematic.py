from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mdtools import dr
import numpy as np

import networkx as nx
from scipy.spatial import cKDTree

from sklearn import datasets, linear_model
from scipy.integrate import cumtrapz

from scratch.sam.util import *



parser = argparse.ArgumentParser('Generate schematic image for given patch pattern')
parser.add_argument('-f', '--infile', default='this_pt.dat', type=str,
                    help='input file name for methyl pt indices (default: %(default)s)')
parser.add_argument('-p', '-ny', default=6, type=int,
                    help='patch dimension p (default %(default)s)')
parser.add_argument('-q', '-nz', default=6, type=int,
                    help='patch dimension q (default %(default)s)')
parser.add_argument('-o', '--outfile', default='schematic.png', type=str,
                    help='output path to write schematic image (default: %(default)s)')

args = parser.parse_args()


this_pt = np.loadtxt(args.infile).astype(int)

state = State(this_pt, ny=args.p, nz=args.q)

state.plot()

plt.savefig('{}'.format(args.outfile), transparent=True)
