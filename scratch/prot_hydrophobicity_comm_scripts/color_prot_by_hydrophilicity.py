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

hydrophobic_sel = "resname ALA or resname VAL or resname LEU or resname ILE or resname PHE or resname TRP or resname PRO"
args = parser.parse_args()

with open(args.charge, 'r') as f:
    charge_assign = pickle.load(f)

sys = MDSystem(args.top, args.struct)

if args.rhodata is not None:
    rho_dat = np.load(args.rhodata)['rho_water'].mean(axis=0)
    sys.find_buried(rho_dat, nb=args.nburied)

sys.assign_hydropathy(charge_assign)
with open(args.charge, 'w') as f:
    pickle.dump(charge_assign, f)

sys.prot.write('prot_by_charge.pdb')
sys.prot_h.write('prot_heavies_by_charge.pdb')
surf_mask = sys.surf_mask
surf = sys.prot[surf_mask]
surf_res = surf.residues
n_surf_residues = surf_res.n_residues
pos_res = surf.select_atoms('resname HIS or resname ARG or resname LYS').residues
neg_res = surf.select_atoms('resname ASP or resname GLU').residues
n_charge_surf_res = pos_res.n_residues + neg_res.n_residues

# Get total abs charge of surface residues
abs_surf_charge = 0
pos_surf_charge = 0
neg_surf_charge = 0
for res in pos_res:
    pos_surf_charge += np.abs(res.atoms.charges.sum())
pos_surf_charge += np.abs(surf.residues[0].atoms.charges.sum())
for res in neg_res:
    neg_surf_charge += np.abs(res.atoms.charges.sum())
neg_surf_charge += np.abs(surf.residues[-1].atoms.charges.sum())

abs_surf_charge = pos_surf_charge + neg_surf_charge

hydrophobic_res = surf.select_atoms(hydrophobic_sel).residues
hydrophilic_res_atoms = surf.select_atoms("not ({})".format(hydrophobic_sel))
hydrophobic_res_atoms = surf.select_atoms(hydrophobic_sel)
n_hydrophilic_res_atoms = hydrophilic_res_atoms.n_atoms
n_hydrophilic_res_atoms_phob = (hydrophilic_res_atoms.tempfactors == 0).sum()

n_hydrophobic_res_atoms = hydrophobic_res_atoms.n_atoms
n_hydrophobic_res_atoms_phob = (hydrophobic_res_atoms.tempfactors == 0).sum()

#embed()

print("Total atoms: {}".format(sys.n_prot_tot))
print("N surf atoms: {}".format(sys.n_surf))
print("  N hydrophilic surf: {}".format(sys.n_phil))
print("  N hydrophobic surf: {}".format(sys.n_phob))
print("  frac hydrophilic surf: {:1.2f}".format(sys.n_phil/sys.n_surf))
print("  frac hydrophobic surf: {:1.2f}".format(sys.n_phob/sys.n_surf))
print("  Abs charge: {:.1f}".format(abs_surf_charge))
print("  Frac abs surf charge: {:1.2e}".format(abs_surf_charge/sys.n_surf))

print("Total Residues: {}".format(sys.prot.residues.n_residues))
print("N surf residues: {}".format(n_surf_residues))
print("  N hydrophobic residues: {}".format(hydrophobic_res.n_residues))
print("  Frac hydrophobic residues: {:1.1f}".format(hydrophobic_res.n_residues/n_surf_residues))
print("  N charged residues: {}".format(n_charge_surf_res))
print("  frac charged residues: {:1.2f}".format(n_charge_surf_res/n_surf_residues))

print("Heavy atoms: {}".format(sys.n_prot_h_tot))
print("N surf: {}".format(sys.n_surf_h))
print("  N hydrophilic: {}".format(sys.n_phil_h))
print("  N hydrophobic: {}".format(sys.n_phob_h))
print("  frac hydrophilic: {}".format(sys.n_phil_h/sys.n_surf_h))

header = "N_tot N_surf N_surf_res N_hydrophilic_surf N_hydrophobic_surf N_hydrophobic_res pos_charge_surf neg_surf_charge N_surf_h  n_phob_h, N_hydrophilic_res_atoms, N_hydrophilic_res_atoms_phob, N_hydrophobic_res_atoms, N_hydrophobic_res_atoms_phob"
out_arr = np.array([sys.n_prot_tot, sys.n_surf, n_surf_residues, sys.n_phil, sys.n_phob, hydrophobic_res.n_residues, 
                    pos_surf_charge, neg_surf_charge, sys.n_surf_h, sys.n_phob_h, n_hydrophilic_res_atoms, n_hydrophilic_res_atoms_phob,
                    n_hydrophobic_res_atoms, n_hydrophobic_res_atoms_phob])


np.savetxt('surf_dat.dat', out_arr, header=header, fmt='%1.2f')
