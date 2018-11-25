from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed
from mdtools import MDSystem
import cPickle as pickle
import argparse


# align smaller atom group to larger
def align(larger, smaller):

    left_offset = 0
    right_offset = 0

    ## Align by assuming something was cut off at beginning or end of larger
    if larger.n_atoms != smaller.n_atoms:
        assert larger.n_atoms > smaller.n_atoms

        while ((larger.resnames[left_offset] != smaller.resnames[0]) or (larger.resnames[left_offset+1] != smaller.resnames[1])):
            left_offset += 1

        while ((larger.resnames[::-1][right_offset] != smaller.resnames[::-1][0]) or (larger.resnames[::-1][right_offset+1] != smaller.resnames[::-1][1])):
            right_offset += 1

        large_mask = np.zeros(larger.n_atoms, dtype=bool)
        large_mask[left_offset:-right_offset] = True

    # Same number of atoms - use all
    else:
        large_mask = np.ones(larger.n_atoms, dtype=bool)

    # Any point mutants?
    pt_mut_mask = larger[large_mask].residues.resnames != smaller.residues.resnames
    
    if pt_mut_mask.sum() > 0:

        larger[large_mask].residues[pt_mut_mask].atoms.tempfactors = -1
        smaller.residues[pt_mut_mask].atoms.tempfactors = -1

    large_mask[larger.tempfactors==-1] = False
    small_mask = smaller.tempfactors != -1

    # Finally, adjust for extra O- atom for smaller 
    if right_offset > 0:
        small_mask[-2:] = False
        large_mask[-right_offset-1] = False

    assert np.array_equal(larger[large_mask].atoms.names, smaller[small_mask].atoms.names)

    return large_mask, small_mask



parser = argparse.ArgumentParser('Find buried atoms, surface atoms (and mask), and dewetted atoms')
parser.add_argument('-s', '--topology', type=str, required=True,
                    help='Input topology file')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Input structure file')
parser.add_argument('--targ-spec', type=str, required=True,
                    help='Selection spec for target heavy atoms. This will be the shape of output array')
parser.add_argument('--part-spec', type=str, required=True,
                    help='Selection spec for partner heavy atoms')
parser.add_argument('--min-dist-targ', type=str, required=True,
                    help='List of min dist of each of this selections atoms to the binding partner')
parser.add_argument('--min-dist-part', type=str, required=True,
                    help='List of min dist of each partner atom to the target')

args = parser.parse_args()

univ = MDAnalysis.Universe(args.topology, args.struct)

univ.add_TopologyAttr('tempfactors')

targ = univ.select_atoms(args.targ_spec)
part = univ.select_atoms(args.part_spec)

print("Target has {} heavy atoms".format(targ.n_atoms))
print("Partner has {} heavy atoms".format(part.n_atoms))

dist_ds_targ = np.load(args.min_dist_targ)
dist_ds_part = np.load(args.min_dist_part)

print("This targ sel spec: {}".format(args.targ_spec))
print("  targ min dist header: {}".format(dist_ds_targ['header']))
print("this part sel spec: {}".format(args.part_spec))
print("  part min dist header: {}".format(dist_ds_part['header']))


dist_targ = dist_ds_targ['min_dist'].mean(axis=0)
dist_part = dist_ds_part['min_dist'].mean(axis=0)

assert dist_targ.shape[0] == targ.n_atoms
assert dist_part.shape[0] == part.n_atoms

if targ.n_atoms >= part.n_atoms:
    targ_mask, part_mask = align(targ, part)
else:
    part_mask, targ_mask = align(part, targ)


## part_mask and targ_mask make sure atoms are aligned - now
##   get union of min distances by taking the min

merged_dist = dist_targ.copy()
merged_dist[targ_mask] = np.min((dist_targ[targ_mask], dist_part[part_mask]), axis=0)

merged_dist = merged_dist[None,:]
np.savez_compressed('min_dist_neighbor_merged.dat', min_dist=merged_dist,
                    header='merging min dist files')

