from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed
from mdtools import MDSystem
import cPickle as pickle
import argparse

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

args = parser.parse_args()

with open(args.charge, 'r') as f:
    charge_assign = pickle.load(f)

sys = MDSystem(args.top, args.struct)

if args.rhodata is not None:
    rho_dat = np.load(args.rhodata)['rho_water'].mean(axis=0)
    sys.find_buried(rho_dat, nb=args.nburied)

sys.assign_hydropathy(charge_assign)

sys.prot.write('prot_by_charge.pdb')
sys.prot_h.write('prot_heavies_by_charge.pdb')
surf_mask = sys.surf_mask
surf = sys.prot[surf_mask]
n_surf_residues = surf.residues.n_residues
charged_res = surf.select_atoms('resname HIS or resname ARG or resname LYS or resname ASP or resname GLU').residues
n_charge_surf_res = charged_res.n_residues

print("Total atoms: {}".format(sys.n_prot_tot))
print("N surf: {}".format(sys.n_surf))
print("  N hydrophilic: {}".format(sys.n_phil))
print("  N hydrophobic: {}".format(sys.n_phob))
print("  frac hydrophilic: {:1.2f}".format(sys.n_phil/sys.n_surf))
print("  frac hydrophobic: {:1.2f}".format(sys.n_phob/sys.n_surf))
print("  n charged residues: {}".format(n_charge_surf_res))
print("  n tot surf resiudes: {}".format(n_surf_residues))
print("  frac charged residues: {:1.2f}".format(n_charge_surf_res/n_surf_residues))

print("Heavy atoms: {}".format(sys.n_prot_h_tot))
print("N surf: {}".format(sys.n_surf_h))
print("  N hydrophilic: {}".format(sys.n_phil_h))
print("  N hydrophobic: {}".format(sys.n_phob_h))
print("  frac hydrophilic: {}".format(sys.n_phil_h/sys.n_surf_h))

