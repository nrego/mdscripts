from __future__ import division, print_function

from constants import k
import os, glob
import numpy as np
import MDAnalysis

import argparse

## Compare actual bound contacts to predicted, then output each, coloring by false postives/False negatives,
#.  and true positives

parser = argparse.ArgumentParser()
parser.add_argument('--contact-mask', default='contact_mask.dat', help='boolean arr of actual contacts')
parser.add_argument('--pred-struct', help='pdb file with precicted contacts', type=str)

args = parser.parse_args()


univ = MDAnalysis.Universe(args.pred_struct)
pred_mask = univ.atoms.tempfactors == 0
contact_mask = np.loadtxt(args.contact_mask, dtype=bool)

assert univ.atoms.n_atoms == contact_mask.size


tp_mask = contact_mask & pred_mask
print("tp: {}".format(tp_mask.sum()))
fp_mask = ~contact_mask & pred_mask
print("fp: {}".format(fp_mask.sum()))
fn_mask = contact_mask & ~pred_mask
print("fn: {}".format(fn_mask.sum()))

univ.atoms.tempfactors = 0
univ.atoms[tp_mask].tempfactors = 1
univ.atoms[fp_mask].tempfactors = 2
univ.atoms.write('pred_tp_fp.pdb')

univ.atoms.tempfactors = 0
univ.atoms[tp_mask].tempfactors = 1
univ.atoms[fn_mask].tempfactors = 2
univ.atoms.write('actual_tp_fn.pdb')

