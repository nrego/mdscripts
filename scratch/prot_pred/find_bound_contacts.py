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

from IPython import embed

# Given bound simulation, find interfacial contact atoms by their dewetting in bound state 
#  (w.r.t unbound unbiased equil simulation)


# Find the normalized water density, exclude any buried atoms (by setting their normed density to 1.0),
#    and then locate dewet atoms that fall below dewetting thresh.

# Returns: tuple of : normed_rho: normalized water density for each atom (after excluding buried atoms)
#                     dewet_mask: mask identifiying dewetted atoms
def find_contact_atoms(avg_water_ref, avg_water_targ, buried_mask, dewetting_thresh=0.5):

    normed_rho = avg_water_targ / avg_water_ref
    normed_rho[buried_mask] = 1.0


    return (normed_rho, (normed_rho < dewetting_thresh) )


parser = argparse.ArgumentParser()
parser.add_argument('--buried_mask', default='buried_mask.dat')
parser.add_argument('--target_bound_struct', help='pdb file of dynamic_water_avg for target in bound state', type=str)
parser.add_argument('--rho_ref', help='rho_data file for unbound, unbiased ref')
parser.add_argument('--rho_targ', help='rho_data file for bound target')

args = parser.parse_args()

embed()

buried_mask = np.loadtxt(args.buried_mask).astype(bool)
target_univ = MDAnalysis.Universe(args.target_bound_struct)

target_avg_water = np.load(args.rho_targ)['rho_water'].mean(axis=0)
ref_avg_water = np.load(args.rho_ref)['rho_water'].mean(axis=0)

assert target_avg_water.size == ref_avg_water.size == target_univ.atoms.n_atoms

find_contact_atoms(ref_avg_water, target_avg_water)

