from __future__ import division, print_function

import MDAnalysis
import numpy as np
import argparse

parser = argparse.ArgumentParser('find native contacts between target and partner')

parser.add_argument('-c', '--struct', required=True, help='structure (GRO or TPR) file')
parser.add_argument('-f', '--traj', required=True, help='trajectory (XTC or TRR) file')
parser.add_argument('-r', '--radius', default=3.8, help='radius to search for contacts')
parser.add_argument('-b', '--start', default=0, type=int, help='start frame')
parser.add_argument('-e', '--end', type=int, help='last frame')
parser.add_argument('--target', type=str, required=True, help='protein for which we get nat contacts')
parser.add_argument('--partner', type=str, required=True, help='binding partner')

args = parser.parse_args()


univ = MDAnalysis.Universe(args.struct, args.traj)

univ.atoms.bfactors = 0 # 1 if contact

first_frame = args.start
if args.end is not None:
    last_frame = args.end
else:
    last_frame = univ.trajectory.n_frames


targ = univ.select_atoms(args.target)
targ_global_indices = targ.indices
targ_local_indices = np.arange(targ.n_atoms)
prot = univ.select_atoms("{} or {}".format(args.target, args.partner))

contact_mask = np.ones((targ.atoms.n_atoms), dtype=bool)

for i_frame in range(first_frame, last_frame):
    ts = univ.trajectory[i_frame]

    if i_frame % 100 == 0:
        print("frame: {}".format(i_frame))
    prot.bfactors = 0
    this_contacts = prot.select_atoms("({}) and around {} ({})".format(args.target, args.radius, args.partner))
    this_contacts.bfactors = 1

    this_contact_mask = targ.bfactors == 1

    contact_mask &= this_contact_mask

ts = univ.trajectory[first_frame]
prot.atoms.bfactors = 1
targ.atoms[contact_mask].bfactors = 0
prot.atoms.segids = 'P'
targ.atoms.segids = 'T'

targ.atoms.write('nat_contacts.pdb')
prot.atoms.write('complex_nat_contacts.pdb')

