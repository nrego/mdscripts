from __future__ import division, print_function
import numpy as np
from scipy.spatial import cKDTree
import MDAnalysis
import argparse
import cPickle as pickle

from mdtools import MDSystem

from IPython import embed

import itertools

k_e = 138.9354859
from constants import k


parser = argparse.ArgumentParser('Analyze bound state of protein-protein complex')
parser.add_argument('-s', '--top', required=True, type=str,
                    help='TOP file')
parser.add_argument('-c', '--struct', required=True, type=str,
                    help='GRO file')
parser.add_argument('--targ-sel-spec', default='segid seg_0_Protein_chain_t')
parser.add_argument('--part-sel-spec', default='segid seg_1_Protein_chain_p')
parser.add_argument('--charge-assign', default='charge_assign.pkl', type=str,
                    help='dictionary with KR hydrophobicities for each atom')
parser.add_argument('--r-cut', default=4.5, type=float,
                    help='Distance of contact pairs')
args = parser.parse_args()

rcut = args.r_cut

targ = MDSystem(args.top, args.struct, sel_spec="segid t")
part = MDSystem(args.top, args.struct, sel_spec="segid p")

with open(args.charge_assign, 'r') as fin:
    charge_assign = pickle.load(fin)

targ.assign_hydropathy(charge_assign)
part.assign_hydropathy(charge_assign)

with open(args.charge_assign, 'w') as fout:
    pickle.dump(charge_assign, fout)

embed()

tree_targ = cKDTree(targ.prot.positions)
tree_targ_h = cKDTree(targ.prot_h.positions)
tree_part = cKDTree(part.prot.positions)
tree_part_h = cKDTree(part.prot_h.positions)

# Shape: (n_targ_atoms, n_part_atoms), [i,j] gives 1 if target 
targ_neigh_list = tree_targ.query_ball_tree(tree_part, r=rcut)
part_neigh_list = tree_part.query_ball_tree(tree_targ, r=rcut)

# Indices of partner atoms that form contacts
part_contacts = itertools.chain(*targ_neigh_list)
part_contacts = np.unique(np.fromiter(part_contacts, int))
targ_contacts = itertools.chain(*part_neigh_list)
targ_contacts = np.unique(np.fromiter(targ_contacts, int))

# Shape: (n_targ_atoms, n_part_atoms), [i,j] gives 1 if target 
targ_h_neigh_list = tree_targ_h.query_ball_tree(tree_part_h, r=rcut)
part_h_neigh_list = tree_part_h.query_ball_tree(tree_targ_h, r=rcut)

# Indices of partner atoms that form contacts
part_h_contacts = itertools.chain(*targ_h_neigh_list)
part_h_contacts = np.unique(np.fromiter(part_h_contacts, int))
targ_h_contacts = itertools.chain(*part_h_neigh_list)
targ_h_contacts = np.unique(np.fromiter(targ_h_contacts, int))

contacts_p_p = []
contacts_np_np = []
contacts_np_p = []

contacts_h_p_p = []
contacts_h_np_np = []
contacts_h_np_p = []

for i_targ in targ_contacts:
    i_phobicity = targ.prot[i_targ].tempfactor
    this_part_neigh = targ_neigh_list[i_targ]
    assert len(this_part_neigh) != 0

    for j_part in this_part_neigh:
        j_phobicity = part.prot[j_part].tempfactor

        contact_phob = i_phobicity + j_phobicity
        if contact_phob == 0:
            contacts_np_np.append((i_targ, j_part))
        elif contact_phob == -1:
            contacts_np_p.append((i_targ, j_part))
        elif contact_phob == -2:
            contacts_p_p.append((i_targ, j_part))
        else:
            raise ValueError

# No just heavy atoms
for i_targ in targ_h_contacts:
    i_phobicity = targ.prot_h[i_targ].tempfactor
    this_part_neigh = targ_h_neigh_list[i_targ]
    assert len(this_part_neigh) != 0

    for j_part in this_part_neigh:
        j_phobicity = part.prot_h[j_part].tempfactor

        contact_phob = i_phobicity + j_phobicity
        if contact_phob == 0:
            contacts_h_np_np.append((i_targ, j_part))
        elif contact_phob == -1:
            contacts_h_np_p.append((i_targ, j_part))
        elif contact_phob == -2:
            contacts_h_p_p.append((i_targ, j_part))
        else:
            raise ValueError


targ_contact_atoms = targ.prot[targ_contacts]
part_contact_atoms = part.prot[part_contacts]

tree_targ_contact = cKDTree(targ_contact_atoms.positions)
tree_part_contact = cKDTree(part_contact_atoms.positions)

# Shape: (n_targ_contacts, n_part_contacts)
d_mat = np.matrix(tree_targ_contact.sparse_distance_matrix(tree_part_contact, max_distance=100).toarray())
r_recip = 1 / d_mat
targ_charges = np.matrix(targ_contact_atoms.charges)
part_charges = np.matrix(part_contact_atoms.charges).T

coul = (beta * k_e * targ_charges * r_recip * part_charges).item()

print("targ contacts (all) : {}  part contacts (all) : {}".format(targ_contact_atoms.n_atoms, part_contact_atoms.n_atoms))
print("NP-NP contacts: {}  P-P contacts: {}  NP-P contacts: {}".format(len(contacts_np_np), len(contacts_p_p), len(contacts_np_p)))
print("coul (kT) : {:0.2f}  coul per atom (kT) : {:0.2f}".format(coul, coul/(targ_contact_atoms.n_atoms+part_contact_atoms.n_atoms)))


# test
#t, p = np.meshgrid(targ_charges, part_charges, indexing='ij')
#coul = t*p*(1/d_mat)
'''
coul = 0.0
for i, targ_atm in enumerate(targ_contact_atoms):
    qi = targ_atm.charge
    for j, part_atm in enumerate(part_contact_atoms):
        dist = d_mat[i,j]
        qj = part_atm.charge
        coul += (qi * qj)/dist
'''