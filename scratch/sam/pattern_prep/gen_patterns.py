from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed

import pickle
from matplotlib import pyplot as plt

from scratch.neural_net import *
from scratch.sam.util import *

import os

## Run *after* enumerating pattern indices with WL (via gen_pt_idx.py)
#.  This takes those output files and generates the actual structure gro files

def generate_pattern(univ, univ_ch3, ids_res):
    univ.atoms.tempfactors = 1
    #embed()
    if ids_res.size == 0:
        return univ
    for idx_res in ids_res:
        res = univ.residues[idx_res]
        ag_ref = res.atoms
        res.atoms.tempfactors = 0

        ch3_res = univ_ch3.residues[idx_res]
        ag_ch3 = ch3_res.atoms

        ch3_shift = ag_ref[-1].position - ag_ch3[-1].position
        ag_ch3.positions += ch3_shift

    newuniv = MDAnalysis.core.universe.Merge(univ.atoms[univ.atoms.tempfactors == 1], univ_ch3.residues[ids_res].atoms)

    univ.atoms.tempfactors = 1

    return newuniv

parser = argparse.ArgumentParser('Generate pattern from list of indices (point data)')
parser.add_argument('-oh', required=True, type=str,
                    help='Structure of oh SAM')
parser.add_argument('-ch3', required=True, type=str,
                    help='Structure of CH3 SAM')
parser.add_argument('--point-data', default='pt_idx_data.pkl', 
                    help='Pickled datafile with pattern indices')
parser.add_argument('--n-samples', default=1, type=int,
                    help='Number of samples to pull for each rms bin (default: %(default)s)')
parser.add_argument('-p', '--patch-size', default=6, type=int,
                    help='Size of patch sides; total size is patch_size**2 (default: 6)')
parser.add_argument('-q', '--patch-size-2', default=None, type=int,
                    help='Size of other patch dimension (default same as patch_size)')
parser.add_argument('--ignore-dir', action='store_true', 
                    help='If true, do not write new d_* directories if one already exists (default: False)')

args = parser.parse_args()

univ_oh = MDAnalysis.Universe(args.oh)
univ_ch3 = MDAnalysis.Universe(args.ch3)
univ_oh.add_TopologyAttr('tempfactors')
univ_ch3.add_TopologyAttr('tempfactors')

assert univ_oh.residues.n_residues == univ_ch3.residues.n_residues

# The last 36 residues are the patch
patch_size_2 = args.patch_size if args.patch_size_2 is None else args.patch_size_2
p = args.patch_size
q = patch_size_2

N = p*q

n_tot_res = univ_oh.residues.n_residues
patch_start_idx = n_tot_res - N

with open(args.point_data, 'rb') as fin:
    bins, occupied_idx, positions, sampled_pt_idx = pickle.load(fin)

bins = bins[0]
n_bins = occupied_idx.sum()
n_samples = args.n_samples

np.random.seed()
print("{} bins occupied, {} samples per bin (up to {} total structures will be generated)".format(n_bins, n_samples, n_bins*n_samples))

#embed()
for bin, pts in zip(bins[occupied_idx], sampled_pt_idx[occupied_idx]):
    # pts should be shape: (n_pts,k)
    n_pts = pts.shape[0]

    # Randomly choose points from this bin
    random = np.random.choice(n_pts, n_pts, replace=False)
    #random=np.arange(n_pts)

    print("{} pts with bin {}".format(n_pts, bin))

    dirname = 'd_{:03d}'.format(int(bin*100))

    flag = True
    i = 1
    while flag:
        try:
            os.makedirs(dirname)
            flag = False
        except FileExistsError:
            if args.ignore_dir:
                print("directory {} already exits - exiting".format(dirname))
                sys.exit()
            else:
                dirname = 'd_{:03d}_{:02d}'.format(int(bin*100), i)
                i += 1

    this_n_sample = min(n_pts, args.n_samples)

    for i_sample in range(this_n_sample):
        try:
            subdir = 'trial_{:d}'.format(i_sample)
            os.makedirs('{}/{}'.format(dirname,subdir))
        except:
            pass


        #with open('{}/{}/this_pt.pkl'.format(dirname, subdir), 'w') as fout:
        #    payload = (positions, this_pt)
        #    pickle.dump(payload, fout)

        this_pt = pts[random][i_sample]
        state = State(this_pt.astype(int), args.patch_size, args.patch_size_2)
        pos_ext = state.pos_ext
        methyl_mask = np.zeros(N).astype(bool)
        if this_pt.size > 0:
            methyl_mask[this_pt] = True

        state.plot()
        plt.savefig('{}/{}/schematic2.pdf'.format(dirname, subdir))

        plt.close('all')

        fig, ax = plt.subplots(figsize=(5.5,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(positions[:,0], positions[:,1], 'bo', markersize=18)
        if this_pt.size > 0:
            this_pos = positions[this_pt]
            ax.plot(this_pos[:,0], this_pos[:,1], 'ko', markersize=18)
        fig.savefig('{}/{}/schematic.pdf'.format(dirname, subdir))
        
        plt.close('all')

        newuniv = generate_pattern(univ_oh, univ_ch3, this_pt+patch_start_idx)
        newuniv.atoms.write('{}/{}/struct.gro'.format(dirname, subdir))

        np.savetxt('{}/{}/this_pt.dat'.format(dirname, subdir), this_pt, fmt='%d')



