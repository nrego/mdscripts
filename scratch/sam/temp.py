from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

import matplotlib as mpl

from matplotlib import pyplot as plt

from scipy.spatial import cKDTree
import os, glob, pathlib
from scratch.neural_net import *
from scratch.sam.util import *

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


univ_oh = MDAnalysis.Universe('whole_oh.gro')
univ_ch3 = MDAnalysis.Universe('whole_ch3.gro')
univ_oh.add_TopologyAttr('tempfactors')
univ_ch3.add_TopologyAttr('tempfactors')

assert univ_oh.residues.n_residues == univ_ch3.residues.n_residues



### Generate inverse patterns for patch_size=4 (N=16)
patch_size = 4
N = patch_size**2
pos_idx = np.arange(N, dtype=int)
n_tot_res = univ_oh.residues.n_residues
patch_start_idx = n_tot_res - N

positions = gen_pos_grid(patch_size)
pos_ext = gen_pos_grid(patch_size+2, z_offset=True, shift_y=-1, shift_z=-1)
d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

#for k in range(1,15):
for k in [15]:

    fnames = sorted(glob.glob('k_{:02d}/d_*/trial_0/this_pt.dat'.format(k)))

    for fname in fnames:
        p = pathlib.Path(fname)
        rms = p.parts[1]

        newpath = 'l_{:02d}/{}/trial_0'.format(k, rms)

        try:
            os.makedirs(newpath)
        except:
            continue
        old_pt = np.loadtxt(fname, ndmin=1).astype(int)
        new_pt = np.setdiff1d(pos_idx, old_pt)

        np.savetxt('{}/this_pt.dat'.format(newpath), new_pt, fmt='%d')

        methyl_mask = np.zeros(N, dtype=bool)
        methyl_mask[new_pt] = True

        ## Plot out pattern
        feat = np.zeros(pos_ext.shape[0])
        feat[patch_indices[methyl_mask]] = 1
        feat[patch_indices[~methyl_mask]] = -1

        ny, nz = patch_size+2, patch_size+2
        this_feat = feat.reshape(ny,nz).T[::-1, :]

        this_feat = this_feat.reshape(1,1,ny,nz)
        plot_hextensor(this_feat)
        plt.savefig('{}/schematic2.pdf'.format(newpath))
        plt.close('all')

        newuniv = generate_pattern(univ_oh, univ_ch3, new_pt+patch_start_idx)
        newuniv.atoms.write('{}/struct.gro'.format(newpath))

