from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed
from mdtools import MDSystem
import pickle
import argparse

from scratch.prot_pred import align

import os, glob

homedir = os.environ['HOME']
dirname = ('{}/simulations/initial_prep'.format(homedir))

u_1wr1 = MDAnalysis.Universe('{}/1wr1_ubiq/dimer/equil.tpr'.format(dirname), '{}/1wr1_ubiq/dimer/cent.gro'.format(dirname))
u_2k6d = MDAnalysis.Universe('{}/2k6d_ubiq/dimer/equil.tpr'.format(dirname), '{}/2k6d_ubiq/dimer/cent.gro'.format(dirname))
u_2qho = MDAnalysis.Universe('{}/2qho_ubiq/dimer/equil.tpr'.format(dirname), '{}/2qho_ubiq/dimer/cent.gro'.format(dirname))
u_2z59 = MDAnalysis.Universe('{}/2z59_ubiq/dimer/equil.tpr'.format(dirname), '{}/2z59_ubiq/dimer/cent.gro'.format(dirname))

u_1wr1.add_TopologyAttr('tempfactors')
u_2k6d.add_TopologyAttr('tempfactors')
u_2qho.add_TopologyAttr('tempfactors')
u_2z59.add_TopologyAttr('tempfactors')

prot_1wr1 = u_1wr1.select_atoms('segid seg_0_Protein_targ and not name H*')
prot_2k6d = u_2k6d.select_atoms('segid seg_1_Protein_chain_B and not name H*')
prot_2qho = u_2qho.select_atoms('segid seg_0_Protein_targ and not name H*')
prot_2z59 = u_2z59.select_atoms('segid seg_1_Protein_targ and not name H*')


min_dist_1wr1 = np.load('{}/1wr1_ubiq/dimer/min_dist_neighbor.dat.npz'.format(dirname))['min_dist'].mean(axis=0)
min_dist_2k6d = np.load('{}/2k6d_ubiq/dimer/min_dist_neighbor.dat.npz'.format(dirname))['min_dist'].mean(axis=0)
min_dist_2qho = np.load('{}/2qho_ubiq/dimer/min_dist_neighbor.dat.npz'.format(dirname))['min_dist'].mean(axis=0)
min_dist_2z59 = np.load('{}/2z59_ubiq/dimer/min_dist_neighbor.dat.npz'.format(dirname))['min_dist'].mean(axis=0)

## Use 2z59 as our base (largest consensus seq)
merged_min_dist = min_dist_2z59.copy()

# w/ 1wr1
main_mask, other_mask = align(prot_2z59, prot_1wr1)

merged_min_dist[main_mask] = np.min((merged_min_dist[main_mask], min_dist_1wr1[other_mask]), axis=0)
del main_mask, other_mask
# w/ 2k6d
other_mask, main_mask = align(prot_2k6d, prot_2z59)

merged_min_dist[main_mask] = np.min((merged_min_dist[main_mask], min_dist_2k6d[other_mask]), axis=0)
del main_mask, other_mask
# w/ 2qho
main_mask, other_mask = align(prot_2z59, prot_2qho)

merged_min_dist[main_mask] = np.min((merged_min_dist[main_mask], min_dist_2qho[other_mask]), axis=0)

merged_min_dist = merged_min_dist[None,:]
np.savez_compressed('min_dist_neighbor_merged.dat', min_dist=merged_min_dist,
                    header='merging min dist files')



