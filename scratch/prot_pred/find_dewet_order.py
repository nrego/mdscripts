from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl

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

parser = argparse.ArgumentParser('Find order of surface atom dewetting from predicted contacts')
parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                    help='Input file names (list of pred_contact_mask.dat)')
parser.add_argument('--buried-mask', required=True, type=str,
                    help='Mask of buried heavy atoms')
parser.add_argument('--actual-contact-mask', required=True, type=str,
                    help='Mask of actual contact heavy atoms')
parser.add_argument('--hydropathy-mask', required=True, type=str,
                    help='Mask of hydropathy of surface atoms')
parser.add_argument('--no-beta', action='store_true',
                    help='If true, assume file names correspond to phi in kJ/mol (default: False, in kT units)')
parser.add_argument('--struct', '-c', required=True, type=str,
                    help='Structure file (just protein heavy atoms) on which to color atoms by phobicity')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.struct)
univ.add_TopologyAttr('tempfactor')
prot = univ.atoms
prot.tempfactors = -2

buried_mask = np.loadtxt(args.buried_mask, dtype=bool)
contact_mask = np.loadtxt(args.actual_contact_mask, dtype=bool)
hydropathy_mask = np.loadtxt(args.hydropathy_mask, dtype=bool)

assert prot.n_atoms == buried_mask.size
surf_mask = ~buried_mask
prot_surf = prot[surf_mask]

fnames = sorted(args.input)
n_files = len(fnames)
atom_dewetting = np.zeros((prot.n_atoms, n_files), dtype=int)
beta_phis = np.array([float(fname.split('/')[0].split('_')[-1]) / 100 for fname in fnames])


for i, fname in enumerate(fnames):
    pred_contact = np.loadtxt(fname, dtype=bool)
    assert pred_contact.size == prot.n_atoms

    # Sanity - all buried atoms should *not* be predicted to be contacts
    assert ~pred_contact[buried_mask].all()

    atom_dewetting[:,i] = pred_contact

surf_atom_dewetting = atom_dewetting[surf_mask]
surf_local_indices = np.arange(surf_atom_dewetting.shape[0])

ii, betas = np.meshgrid(surf_local_indices, beta_phis, indexing='ij')

# shape (n_surf_atoms, ): gives beta phi * for each
perm_dewetting = np.zeros_like(surf_local_indices).astype(float)
# shape (n_surf_atoms, ): beta_phi at which atom first dewets
first_dewetting = np.zeros_like(perm_dewetting).astype(float)
for atom_i in surf_local_indices:
    atom_i_dewet = surf_atom_dewetting[atom_i]
    low_phi_dewet = 4.0
    for j, beta in enumerate(beta_phis):
        if beta == beta_phis[-1]:
            perm_dewetting[atom_i] = beta_phis[-1]
            first_dewetting[atom_i] = beta_phis[-1]
        if atom_i_dewet[j] and beta < low_phi_dewet:
            low_phi_dewet = beta
        if atom_i_dewet[j:].all():
            perm_dewetting[atom_i] = beta
            first_dewetting[atom_i] = low_phi_dewet
            break

prot_surf.tempfactors = perm_dewetting
prot.write('prot_atoms_dewetting_order.pdb')
prot_surf.tempfactors = first_dewetting
prot.write('prot_atoms_first_dewetting_order.pdb')

# Fraction of phobic atoms that are have beta_phi* == beta_phi
marginal_phobicity = np.zeros_like(beta_phis)
# Fraction of phobic atoms that have beta_phi* <= beta_phi
total_phobicity = np.zeros_like(beta_phis)
for i, beta_phi in enumerate(beta_phis):
    marginal_mask = perm_dewetting == beta_phi
    total_mask = perm_dewetting <= beta_phi

    marginal_phobicity[i] = hydropathy_mask[surf_mask][marginal_mask].sum() / marginal_mask.sum()
    total_phobicity[i] = hydropathy_mask[surf_mask][total_mask].sum() / total_mask.sum()

dat = np.vstack((beta_phis, marginal_phobicity, total_phobicity)).T
np.savetxt('phobicity_with_betaphi.dat', dat, fmt='%1.2f')



# Fraction of phobic atoms that are have beta_phi* == beta_phi
marginal_phobicity = np.zeros_like(beta_phis)
# Fraction of phobic atoms that have beta_phi* <= beta_phi
total_phobicity = np.zeros_like(beta_phis)
for i, beta_phi in enumerate(beta_phis):
    marginal_mask = first_dewetting == beta_phi
    total_mask = first_dewetting <= beta_phi

    marginal_phobicity[i] = hydropathy_mask[surf_mask][marginal_mask].sum() / marginal_mask.sum()
    total_phobicity[i] = hydropathy_mask[surf_mask][total_mask].sum() / total_mask.sum()

dat = np.vstack((beta_phis, marginal_phobicity, total_phobicity)).T
np.savetxt('first_phobicity_with_betaphi.dat', dat, fmt='%1.2f')

