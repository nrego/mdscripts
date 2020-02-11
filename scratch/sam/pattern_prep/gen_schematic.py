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
parser.add_argument('-f', '--indirs', default='*/d_*/trial_0', type=str,
                    help='String wildcard expr for listing all the input directories (default: %(default)s)')
parser.add_argument('-p', '-ny', default=6, type=int,
                    help='patch dimension p (default %(default)s)')
parser.add_argument('-q', '-nz', default=6, type=int,
                    help='patch dimension q (default %(default)s)')


args = parser.parse_args()

indirs = sorted(glob.glob(args.indirs))

for i, this_dir in enumerate(indirs):

    print('Doing dir: {} ({:03d} of {:03d})'.format(this_dir, i, len(indirs)))
    this_pt = np.loadtxt('{}/this_pt.dat'.format(this_dir)).astype(int)

    state = State(this_pt, ny=args.p, nz=args.q)

    state.plot()
    plt.savefig('{}/schematic'.format(this_dir))

