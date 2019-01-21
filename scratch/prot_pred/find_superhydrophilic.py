from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from IPython import embed
from mdtools import MDSystem
import cPickle as pickle
import argparse

import os

from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

homedir = os.environ['HOME']


mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':20})


parser = argparse.ArgumentParser('analyze atomic composition of TPs, FPs, etc')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='Topology (TPR) file')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Structure (GRO) file')
parser.add_argument('--sel-spec', type=str, default='segid seg_0_Protein_targ and not name H*',
                    help='Selection spec to get target heavy atoms')
parser.add_argument('--buried', type=str, default='../../bound/buried_mask.dat',
                    help='Buried mask (Default: %(default)s)')
parser.add_argument('--pred-contact', type=str, default='pred_contact_mask.dat',
                    help='Predicted contact mask (Default: %(default)s)')
parser.add_argument('--hydropathy', type=str, default='../../bound/hydropathy_mask.dat',
                    help='Hydropathy mask (Default: %(default)s)')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.top, args.struct)
prot = univ.select_atoms(args.sel_spec)
univ.add_TopologyAttr('tempfactors')

buried_mask = np.loadtxt(args.buried, dtype=bool)
surf_mask = ~buried_mask

pred_contact_mask = np.loadtxt(args.pred_contact, dtype=bool)

hydropathy_mask = np.loadtxt(args.hydropathy, dtype=bool)

prot.tempfactors = -2
super_hydrophil_mask = surf_mask & ~pred_contact_mask
prot[super_hydrophil_mask].tempfactors = -1

prot[super_hydrophil_mask & hydropathy_mask].tempfactors = 1

prot.write('wet.pdb', bonds=None)


