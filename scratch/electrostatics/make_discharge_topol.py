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

parser = argparse.ArgumentParser("Regenerate topology file for a solute, setting all q=0")
parser.add_argument('-s', '--top', default='top.tpr', type=str,
                    help='topology file')
parser.add_argument('-c', '--gro', default='equil.gro', type=str,
                    help='structure file')
parser.add_argument('--sel-spec', required=True, type=str, default='protein',
                    help='Selection spec for (all) protein atoms')
parser.add_argument('--top-file', required=True, type=str,
                    help='Topology file for protein; will replace each interface atom\'s partial charge with 0')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.top, args.gro)
univ.add_TopologyAttr('tempfactors')

prot = univ.select_atoms(args.sel_spec)
prot_h = prot.select_atoms('not name H*')

# In case protein's global first index is not 1
start_idx = prot.atoms[0].id



with open(args.top_file, 'r') as fin:
    lines = fin.readlines()


done_with_atoms = False

fmtstr = '{:>6s}{:>11s}{:>7s}{:>7s}{:>7s}{:>7s}{:>11.4f}{:>11s}{:>15s}'

with open('topol_prot_noq.top', 'w') as fout:
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


        try:
            blah, atm_type, res_index, resname, atm_name, chggrp, charge, mass, _, _, _ = line.split()
        except ValueError:
            blah, atm_type, res_index, resname, atm_name, chggrp, charge, mass = line.split()

        fout.write(fmtstr.format(blah, atm_type, res_index, resname, atm_name, chggrp, 0.0, mass, '; XXX\n'))

