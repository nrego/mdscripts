# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm
from matplotlib import pyplot as plt

import numpy as np

from mdtools import MDSystem

import argparse

from IPython import embed

parser = argparse.ArgumentParser('color protein atoms by their hydrophilicity, '
                                 'optionally excluding buried atoms or applying mask')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='Topology input file (tpr or gro file; need tpr to color buried hydrogens)')
parser.add_argument('-c', '--struct', type=str,
                    help='Structure file (gro, pdb, etc) - output will have same positions')
parser.add_argument('-p', '--charge', type=str, default='charge_assign.pkl',
                    help='pickled double dictionary of [res][atom_type]: hydrophilicity; '
                         'unknown atom types will prompt user')
parser.add_argument('--rhodata', type=str,
                    help='Optionally supply rho data file, for determining buried atoms')
parser.add_argument('-nb', '--nburied', type=float, default=5,
                    help='If rhodata supplied, atoms are considered buried if they have fewer than '
                         'this many average water molecules')
parser.add_argument('--sel-spec', type=str, default='segid targ',
                    help='selection spec for finding protein (all) atoms')


args = parser.parse_args()

sys = MDSystem(args.top, args.struct, sel_spec=args.sel_spec)

rho_dat = np.load(args.rhodata)['rho_water'].mean(axis=0)
max_val = np.ceil(rho_dat.max()) + 1
bb = np.arange(0, max_val, 1)
hist, bb = np.histogram(rho_dat, bins=bb)
#plt.bar(bb[:-1], hist)
#plt.show()

sys.find_buried(rho_dat, args.nburied)

print("N Heavy Atoms: {}".format(sys.n_prot_h_tot))
print("N Surface Heavy (rho>{}): {}".format(args.nburied, sys.n_surf_h))
print("N Buried Heavy: {}".format(sys.n_buried_h))

print("  Reduction by excluding buried: {:0.2f}".format(sys.n_buried_h/sys.n_prot_h_tot))


surf_mask = sys.surf_mask_h
buried_mask = ~surf_mask

np.savetxt('surf_mask.dat', surf_mask, fmt='%1d')
np.savetxt('buried_mask.dat', buried_mask, fmt='%1d')

surf_atoms = sys.prot_h[surf_mask]

print("Writing umbr.conf...")

header_string = "; Umbrella potential for a spherical shell cavity\n"\
"; Name    Type          Group  Kappa   Nstar    mu    width  cutoff  outfile    nstout\n"\
"hydshell dyn_union_sph_sh   OW  0.0     0   XXX    0.01   0.02   phiout.dat   50  \\\n"

with open('umbr.conf', 'w') as fout:
    fout = open('umbr.conf', 'w')
    fout.write(header_string)

    for atm in surf_atoms:
        fout.write("{:<10.1f} {:<10.1f} {:d} \\\n".format(-0.5, 0.6, atm.index+1))

print("...Done.")

