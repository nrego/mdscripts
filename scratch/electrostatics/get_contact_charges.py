from __future__ import division, print_function

import numpy as np
from IPython import embed

import MDAnalysis

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt

import argparse
import os, glob

from mdtools import dr

from constants import k


splitter = lambda s: [float(val) for val in s.split(',')]

parser = argparse.ArgumentParser("Output partial charges of all contact atoms (including hydrogens)")
parser.add_argument('-s', '--top', default='top.tpr', type=str,
                    help='topology file')
parser.add_argument('-c', '--gro', default='equil.gro', type=str,
                    help='structure file')
parser.add_argument('--contact-mask', default='actual_contact_mask.dat', type=str,
                    help='actual_mask')
parser.add_argument('--sel-spec', required=True, type=str,
                    help='Selection spec for protein atoms')
parser.add_argument('--top-file', required=True, type=str,
                    help='Topology file for protein; will replace each interface atom\'s partial charge with scaled charge')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.top, args.gro)
univ.add_TopologyAttr('tempfactors')

prot = univ.select_atoms(args.sel_spec)
prot_h = prot.select_atoms('not name H*')

contact_mask = np.loadtxt(args.contact_mask, dtype=bool)
contacts_h = prot_h[contact_mask]

# In case protein's global first index is not 1
start_idx = prot.atoms[0].id

contact_atom_charges = {}
for contact_atm in contacts_h:
    contact_atm.tempfactor = 1
    contact_atom_charges[contact_atm.id+1-start_idx] = contact_atm.charge

    for atm in contact_atm.bonded_atoms:
        if atm.name[0] != 'H':
            continue
        atm.tempfactor = 1
        contact_atom_charges[atm.id+1-start_idx] = atm.charge

prot.write('contacts.pdb')

outdata = np.zeros((len(contact_atom_charges.items()), 2))
for i, (k, v) in enumerate(contact_atom_charges.iteritems()):

    outdata[i,0] = k
    outdata[i,1] = v

sort_idx = np.argsort(outdata[:,0])

np.savetxt('contact_atom_charges.dat', outdata[sort_idx], fmt='%d %0.4f', header='atom_index  q_i')

##############################################
## Write out new umbrella file for contact heavy atoms
with open('umbr_contact.conf', 'w') as fout:
    header_string = "; Umbrella potential for a spherical shell cavity\n"\
    "; Name    Type          Group  Kappa   Nstar    mu    width  cutoff  outfile    nstout\n"\
    "hydshell dyn_union_sph_sh   OW  0.0     0   XXX    0.01   0.02   phiout.dat   50  \\\n"
    fout.write(header_string)
    for atm in contacts_h:
        fout.write("{:<10.1f} {:<10.1f} {:d} \\\n".format(-0.5, 0.6, atm.index+1))
#############################################


##############################################
## Make new topology file
with open(args.top_file, 'r') as fin:
    lines = fin.readlines()


done_with_atoms = False

fmtstr = '{:>6s}{:>11s}{:>7s}{:>7s}{:>7s}{:>7s}{:>11.4f}{:>11s}{:>7s}{:>11.4f}{:>15s}'

with open('topol_prot_lam.itp', 'w') as fout:
    for i, line in enumerate(lines):

        if done_with_atoms:
            fout.write(line)
            continue

        split = line.split()

        try:
            index = split[0]
        except IndexError:
            fout.write(line)
            continue

        try:
            index = split[1]
        except IndexError:
            fout.write(line)
            continue

        if split[1] == 'bonds':
            done_with_atoms = True
            fout.write(line)
            continue

        try:
            index = int(split[0])
        except ValueError:
            fout.write(line)
            continue

        ## Check if atom index ##
        if index not in contact_atom_charges.keys():
            fout.write(line)

        else:
            try:
                blah, atm_type, res_index, resname, atm_name, chggrp, charge, mass, _, _, _ = line.split()
            except ValueError:
                blah, atm_type, res_index, resname, atm_name, chggrp, charge, mass = line.split()
            assert abs(float(charge) - contact_atom_charges[index]) < 1e-7

            fout.write(fmtstr.format(blah, atm_type, res_index, resname, atm_name, chggrp, float(charge), mass, atm_type, 0.0000, '; XXX\n'))

