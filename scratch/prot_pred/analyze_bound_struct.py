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

from scipy.spatial import cKDTree

from constants import k
beta = 1 / (k*300)

homedir = os.environ['HOME']


mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':20})


## Analyze composition of classified predictions

parser = argparse.ArgumentParser('analyze bound structure, including energetics')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='Topology (TPR) file')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Structure (GRO) file')
parser.add_argument('--targ-spec', type=str, default='segid targ',
                    help='Selection spec to get target heavy atoms')
parser.add_argument('--part-spec', type=str, default='segid B')
parser.add_argument('--actual-contact', type=str, default='../../bound/actual_contact_mask.dat',
                    help='Actual contacts (Default: %(default)s)')
parser.add_argument('--buried', type=str, default='../../bound/buried_mask.dat',
                    help='Buried mask (Default: %(default)s)')
parser.add_argument('--hydropathy', type=str, default='../../bound/hydropathy_mask.dat',
                    help='Hydropathy mask (Default: %(default)s)')
args = parser.parse_args()

sys_targ = MDSystem(args.top, args.struct, sel_spec=args.targ_spec)
sys_part = MDSystem(args.top, args.struct, sel_spec=args.part_spec)

buried_mask = np.loadtxt(args.buried, dtype=bool)
surf_mask = ~buried_mask
hydropathy_mask = np.loadtxt(args.hydropathy, dtype=bool)[surf_mask]
contact_mask = np.loadtxt(args.actual_contact, dtype=bool)[surf_mask]

targ = sys_targ.prot
targ_h = sys_targ.prot_h

assert targ_h.n_atoms == buried_mask.size

sys_targ.find_buried(surf_mask, nb=0.5)

assert np.array_equal(sys_targ.surf_mask_h, surf_mask)

part = sys_part.prot
part_h = sys_part.prot_h
assert part.n_atoms > 0

print("Analyzing interfacial atoms and residues on target protein...")

# Surface heavy atoms on the target that form interfacial contacts
targ_contact_atoms = targ_h[surf_mask][contact_mask]
n_contacts = targ_contact_atoms.n_atoms

n_phob_contacts = (hydropathy_mask & contact_mask).sum()
n_phil_contacts = (~hydropathy_mask & contact_mask).sum()

print("    ...N heavy atom contacts: {}    n_phob: {} ({:0.2f})    n_phil: {} ({:0.2f})".format(n_contacts, n_phob_contacts, n_phob_contacts/n_contacts, n_phil_contacts, n_phil_contacts/n_contacts))

# Their corresponding contact residues
targ_contact_residues = targ_contact_atoms.residues
print("")
print("    ...N contact residues: {}".format(targ_contact_residues.n_residues))

pos_charge = 0
neg_charge = 0
net_charge = 0
abs_charge = 0
charges = np.array([])
for res in targ_contact_residues:
    this_charge = np.round(res.atoms.charges.sum()).astype(int)

    charges = np.append(charges, this_charge)

    if this_charge > 0:
        pos_charge += this_charge
    elif this_charge < 0:
        neg_charge += np.abs(this_charge)

    net_charge += this_charge
    abs_charge += np.abs(this_charge)

print("    pos_charge: +{}    neg_charge: {}   net_charge: {}   abs total charge: {}".format(pos_charge, -neg_charge, net_charge, abs_charge))

tree_targ = cKDTree(targ.positions)
tree_part = cKDTree(part.positions)

r_ij = tree_targ.sparse_distance_matrix(tree_part, max_distance=100)

ecoul = 0.0
evdw = 0.0

# Coul const in (kJ*nm)/(mol*e^2)
k_e = 138.935485
for i_targ in range(targ.n_atoms):
    q_i = targ[i_targ].charge
    for j_part in range(part.n_atoms):

        q_j = part[j_part].charge
        d_ij = r_ij[i_targ, j_part] / 10.0

        ecoul += k_e * (q_i * q_j) / d_ij

print("E_coul (kT): {}".format(beta * ecoul))
print("per_atom coul (kT/atom): {}".format((beta*ecoul)/(targ.n_atoms+part.n_atoms)))





