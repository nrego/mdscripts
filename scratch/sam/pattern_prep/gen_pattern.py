
import numpy as np
import MDAnalysis

import argparse
from IPython import embed

import pickle
from matplotlib import pyplot as plt

from scratch.neural_net import *
from scratch.sam.util import *

from scipy.spatial import cKDTree

import os,sys

## Run *after* enumerating pattern indices with WL (via gen_pt_idx.py)
#.  This takes those output files and generates the actual structure gro files

# Calculate the positions of sp3 hydrogens, given a bond-vector from a backbone
#  to the terminal CT
# H's are placed with bond lengths=1.09 A, and tetrahedral geom (angle 109.5)
def calc_h_pos(p0, vec):

    # CT-HC bond length
    dh = 1.09
    alpha = (109.5 - 90)*(np.pi/180)

    v1 = p0 + np.array([dh*np.cos(alpha), dh*np.sin(alpha), 0])

parser = argparse.ArgumentParser('Generate pattern from list of indices (point data)')
parser.add_argument('--sam-oh', type=str, default='../../struct_oh.gro',
                    help='Structure of pure OH SAM')
parser.add_argument('--pt-idx', type=str, default='this_pt.dat',
                    help='Methyl patch indices (local indices, max is p*q-1)')
parser.add_argument('--p', '--patch-size', default=6, type=int,
                    help='Size of patch sides; total size is patch_size**2 (default: 6)')
parser.add_argument('--q', '--patch-size-2', default=None, type=int,
                    help='Size of other patch dimension (default same as patch_size)')

args = parser.parse_args()


p = args.p
q = args.q
if q is None:
    q = p

pt_idx = np.loadtxt(args.pt_idx, ndmin=1).astype(int)
state = State(pt_idx, p=p, q=q)

plt.close()
state.plot()
plt.savefig("schematic.png", transparent=True)
plt.close()

n_patch = p*q

univ = MDAnalysis.Universe(args.sam_oh)
## So we can mark which OH SAM residues we'll be replacing with CH3s...
univ.add_TopologyAttr("tempfactors")

# OH patch residues
sam_res = univ.residues[-n_patch:]
assert sam_res.atoms.select_atoms("name O12").n_atoms == n_patch

## All non-patch atoms
ag_nonpatch = univ.residues[:-n_patch].atoms


## Total number of endgroups we'll replace
n_replace = pt_idx.size

if n_replace == 0:
    print("Pure OH patch...exiting")
    univ.atoms.write("struct.gro")
    sys.exit()


## There are 15 atoms in each CH3 res
res_map = []
for i in range(n_replace):
    res_map.append(np.ones(15)*i)
res_map = np.concatenate(res_map)


## Univ of methyl groups
univ_replace = MDAnalysis.Universe.empty(n_replace*15, n_replace, atom_resindex=res_map, trajectory=True)
univ_replace.add_TopologyAttr('names')
univ_replace.add_TopologyAttr('resnames')
univ_replace.add_TopologyAttr('resids')
univ_replace.residues.resnames = "CH3"

# i is local index of each patch group that will become a methyl
for i in pt_idx:
    

    this_oh_res = sam_res[i]
    this_ch_res = univ_replace.residues[i]

    this_ch_res.atoms.names = np.append(['HC1', 'HC2', 'HC3', 'CT'], this_oh_res.atoms.names[2:])

    # Positions of new CH3 residue - heavies get same as OH res that's being replaced.
    res_pos = np.zeros((15, 3))
    res_pos[3:]  = this_oh_res.atoms.positions[1:]
    p0 = res_pos[3]
    res_pos[0] = p0 + np.array([1,0,0])
    res_pos[1] = p0 + np.array([0,1,0])
    res_pos[2] = p0 + np.array([0,0,1])

    this_ch_res.atoms.positions = res_pos

    # Mark them to remove
    this_oh_res.atoms.tempfactors = 1

# Remaining OH patch groups
ag_oh = sam_res.atoms[sam_res.atoms.tempfactors != 1]

if ag_oh.n_atoms > 0:
    final_univ = MDAnalysis.core.universe.Merge(ag_nonpatch, ag_oh, univ_replace.atoms)
else:
    final_univ = MDAnalysis.core.universe.Merge(ag_nonpatch, univ_replace.atoms)

final_univ.atoms.write("struct.gro")

