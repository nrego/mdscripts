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

import argparse

from util import find_dewet_atoms



parser = argparse.ArgumentParser()
parser.add_argument('--buried-mask', default='buried_mask.dat')
parser.add_argument('--target-bound-struct', help='pdb file of dynamic_water_avg for target in bound state', type=str)
parser.add_argument('--rho-ref', help='rho_data file for unbound, unbiased ref')
parser.add_argument('--rho-targ', help='rho_data file for bound target')

args = parser.parse_args()

buried_mask = np.loadtxt(args.buried_mask).astype(bool)
target_univ = MDAnalysis.Universe(args.target_bound_struct)

target_avg_water = np.load(args.rho_targ)['rho_water'].mean(axis=0)
ref_avg_water = np.load(args.rho_ref)['rho_water'].mean(axis=0)

assert target_avg_water.size == ref_avg_water.size == target_univ.atoms.n_atoms

norm_rho_bound, contact_mask = find_dewet_atoms(ref_avg_water, target_avg_water, buried_mask)

target_univ.atoms.bfactors = norm_rho_bound
target_univ.atoms.write('bound_norm_atoms.pdb')

target_univ.atoms.bfactors = 1
target_univ.atoms[contact_mask].bfactors = 0
target_univ.atoms.write('bound_dewet_atoms.pdb')

np.savetxt('contacts_dewet.dat', contact_mask, fmt='%1d')
