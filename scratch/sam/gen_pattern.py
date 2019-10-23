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
parser.add_argument('--patch-size', default=6, type=int,
                    help='Size of patch (number of headgroups, default: 36)')

args = parser.parse_args()

univ_oh = MDAnalysis.Universe(args.oh)
univ_ch3 = MDAnalysis.Universe(args.ch3)
univ_oh.add_TopologyAttr('tempfactors')
univ_ch3.add_TopologyAttr('tempfactors')

assert univ_oh.residues.n_residues == univ_ch3.residues.n_residues

# The last 36 residues are the patch
N = args.patch_size**2
n_tot_res = univ_oh.residues.n_residues
patch_start_idx = n_tot_res - N

with open(args.point_data, 'rb') as fin:
    bins, occupied_idx, positions, sampled_pt_idx = pickle.load(fin)
pos_ext = gen_pos_grid(args.patch_size+2, z_offset=True, shift_y=-1, shift_z=-1)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)


n_bins = occupied_idx.sum()
n_samples = args.n_samples

print("{} bins occupied, {} samples per bin ({} total structures will be generated)".format(n_bins, n_samples, n_bins*n_samples))

for bin, pts in zip(bins[:-1][occupied_idx], sampled_pt_idx[occupied_idx]):
    # pts should be shape: (n_pts,k)
    n_pts = pts.shape[0]

    #random = np.random.choice(n_pts, n_pts, replace=False)
    random=np.arange(n_pts)

    print("{} pts with bin {}".format(n_pts, bin))

    dirname = 'd_{:03d}'.format(int(bin*100))
    try:
        os.makedirs(dirname)
    except OSError:
        print("directory {} already exits - exiting".format(dirname))

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
        methyl_mask = np.zeros(N).astype(bool)
        if this_pt.size > 0:
            methyl_mask[this_pt] = True

        feat = np.zeros(pos_ext.shape[0])
        feat[patch_indices[methyl_mask]] = 1
        feat[patch_indices[~methyl_mask]] = -1

        ny, nz = args.patch_size+2, args.patch_size+2
        this_feat = feat.reshape(ny,nz).T[::-1, :]

        this_feat = this_feat.reshape(1,1,ny,nz)
        plot_hextensor(this_feat)
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



