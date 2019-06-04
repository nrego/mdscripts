from __future__ import division, print_function
import numpy as np
from scipy.spatial import cKDTree
import MDAnalysis
import argparse
from IPython import embed

from mdtools import MDSystem

parser = argparse.ArgumentParser('Output protein and partner as pdb file')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='TPR file')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='GRO file')
parser.add_argument('--targ-sel-spec', type=str, required=True,
                    help='Selection for target atoms')
parser.add_argument('--part-sel-spec', type=str, required=True,
                    help='Selection for partner atoms')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.top, args.struct)
targ = univ.select_atoms('{}'.format(args.targ_sel_spec))
part = univ.select_atoms('{}'.format(args.part_sel_spec))
prot = univ.select_atoms('({}) or ({})'.format(args.targ_sel_spec, args.part_sel_spec))

for seg in targ.segments:
    seg.segid = 'targ'
for seg in part.segments:
    seg.segid = 'part'

targ.atoms.ids = np.arange(targ.n_atoms)+1
targ.residues.resids = np.arange(targ.residues.n_residues) + 1

part.atoms.ids = np.arange(part.n_atoms)+1
part.residues.resids = np.arange(part.residues.n_residues) + 1

prot.write('prot.pdb', bonds=None)
