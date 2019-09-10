from __future__ import division, print_function

import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

from constants import k

import MDAnalysis

import argparse

from scipy.spatial import cKDTree

top = MDAnalysis.Universe("test/top.tpr")
univ = MDAnalysis.Universe("bound/actual_contact.pdb")

buried_mask = np.loadtxt("pred_reweight/beta_phi_000/buried_mask.dat", dtype=bool)
surf_mask = ~buried_mask
hydropathy_mask = np.loadtxt("bound/hydropathy_mask.dat", dtype=bool)[surf_mask]
contact_mask = np.loadtxt("bound/actual_contact_mask.dat", dtype=bool)[surf_mask]
beta_phi_star = np.loadtxt("beta_phi_star_prot.dat")[surf_mask]

surf_atoms = univ.atoms[surf_mask]
surf_charge = top.select_atoms("not name H*")[surf_mask]
n_atoms = surf_atoms.n_atoms
tree = cKDTree(surf_atoms.positions)

neighbors = tree.query_ball_tree(tree, r=6.0)

for i, neigh in enumerate(neighbors):
    idx = np.where(np.array(neigh)==i)[0].item()
    neighbors[i] = np.delete(neigh, idx)

# Fraction of atom i's neighbors that are phobic
neigh_phob = np.zeros(n_atoms)
# Total number of phobic neighbors
neigh_phob_sum = np.zeros(n_atoms)
# Average beta phi star of neighbors
neigh_bphi = np.zeros(n_atoms)

for i, neigh in enumerate(neighbors):
    n_neigh = neigh.size
    if n_neigh == 0:
        continue

    neigh_phob[i] = hydropathy_mask[neigh].mean()
    neigh_phob_sum[i] = hydropathy_mask[neigh].sum()
    neigh_bphi[i] = beta_phi_star[neigh].mean()
