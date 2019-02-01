from __future__ import division

import numpy as np
import MDAnalysis

import argparse
import glob, os

import cPickle as pickle

def find_idx_from_struct(struct):
    univ = MDAnalysis.Universe(struct)

    patch_res = univ.residues[-36:]

    n_tot_res = univ.residues.n_residues
    patch_start_idx = n_tot_res - 36

    indices = []
    for res in patch_res:
        if res.resname == 'CH3':
            indices.append(res.resid - patch_start_idx - 1)

    return indices


fnames = glob.glob('k_*/d_*/trial_0/struct.gro')

for fname in fnames:
    print("fname: {}".format(fname))
    subdir = os.path.dirname(fname)

    indices = find_idx_from_struct(fname)

    np.savetxt('{}/this_pt.dat'.format(subdir), indices, fmt='%2d')