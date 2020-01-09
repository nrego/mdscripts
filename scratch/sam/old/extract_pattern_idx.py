from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed

import cPickle as pickle
from matplotlib import pyplot as plt

import os


parser = argparse.ArgumentParser('Extract point index (indices) from pattern')
parser.add_argument('-struct', type=str, default='struct.gro',
                    help='Structure of patterned SAM')
parser.add_argument('--point-data', default='../../pt_idx_data.pkl', 
                    help='Pickled datafile with pattern indices')
args = parser.parse_args()

try:
    indices = np.loadtxt('this_pt.dat', dtype=int)
except:
    indices = None

univ = MDAnalysis.Universe(args.struct)
res = univ.residues[-36:]

patch_start_idx = univ.residues.n_residues - 36

with open(args.point_data, 'r') as fin:
    rms_bins, occupied_idx, positions, sampled_pt_idx = pickle.load(fin)

patch_ch3_indices = []

for patch_res in res:
    if patch_res.resname != 'CH3':
        continue

    patch_ch3_indices.append(patch_res.resid-patch_start_idx-1)

patch_ch3_indices = np.sort(patch_ch3_indices).squeeze()

if indices is not None:
    assert np.array_equal(indices, patch_ch3_indices)

if patch_ch3_indices.ndim == 0:
    patch_ch3_indices = patch_ch3_indices.reshape(1,)
np.savetxt('this_pt.dat', patch_ch3_indices, fmt='%d')