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
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Dewetting threshold for normalized rho (default: 0.5)')
parser.add_argument('--sel-spec', default='segid targ', type=str,
                    help='Selection spec for getting protein atoms')

args = parser.parse_args()

sys = MDSystem(args.topology, args.struct, sel_spec=args.sel_spec)

ref_data = np.load(args.ref)['rho_water'].mean(axis=0)
targ_data = np.load(args.rhodata)['rho_water'].mean(axis=0)

sys.find_buried(ref_data, nb=args.nb)

# Surface heavy atoms
surf_mask = sys.surf_mask_h
buried_mask = ~surf_mask
prot = sys.prot_h


rho_i = targ_data / ref_data
contact_mask = (rho_i < args.thresh) & surf_mask
prot[contact_mask].tempfactors = 1

np.savetxt('contact_mask.dat', contact_mask, fmt='%1d')

prot.write('contact.pdb')

prot.tempfactors = rho_i
prot[buried_mask].tempfactors = -2
prot.write('contact_rho.pdb')
