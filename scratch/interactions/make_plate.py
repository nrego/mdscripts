from __future__ import division, print_function; __metaclass__ = type
import numpy as np

import MDAnalysis

import argparse
from util import *

from scipy.spatial import cKDTree

from matplotlib import pyplot as plt

from IPython import embed



# head group names
# C: non-polar group (methyl), D: polar group (h-bond donor), N: negative group, P: positive group
atom_head_names = ['CH3', 'OH', 'N', 'P']
atom_mass = {'CH3': 12.01,
             'OH':  16.0,
             'BOH': 16.0,
             'HO':  1.008,
              'N': 15.9994,
              'P': 15.9994}
atom_charge = {'CH3': 0.0,
             'OH':  -0.48936,
             'BOH': -0.48936,
             'HO':   0.48936,
              'N':  -1.0,
              'P':   1.0}
res_names = ['CH3', 'OH', 'NEG', 'POS']

# Indices of adjacent slices (vsites) to restrain with bonds
slice_pairs = [(0,1), (0,4), (1,5), (2,3), (2,4), (3,5)]


parser = argparse.ArgumentParser("Construct hex-packed plate system, given number of central atoms. Also build topology")
parser.add_argument("--n-mid", default=7, type=int,
                    help="Number of middle atoms (default: %(default)s)")
parser.add_argument("--pattern", type=str, default=None,
                    help="filename of file detailing surface pattern (should be readable as numpy array,"\
                         "each entry indicates type of headgroup for that position; e.g. 0 for ch3, 1 for oh, 2 for neg, and 3 for pos")
parser.add_argument("--d-space", "--d", type=float, default=3.0,
                    help="Lattice spacing (Angstroms) between atoms (default: %(default)s)")
parser.add_argument("--rcut", "-rc", type=float, default=1.0,
                    help="Cut-off for pair-wise interactions")
parser.add_argument("--idx-offset", type=int, default=0,
                    help="shift indices by this offset")
args = parser.parse_args()

n_vsites = 0
n_mid = args.n_mid
d = args.d_space
rcut = args.rcut

n_res = n_mid + 2*fib(n_mid, n_mid)

if args.pattern is not None:
    pattern = np.loadtxt(args.pattern, dtype=int)
else:
    pattern = np.zeros(n_res, dtype=int)

assert pattern.size == n_res

positions, center_pt_idx, center_indices, edge_indices, vert_slice, slices = gen_plate_position(n_mid, d)

## make an anotated schematic of our plate
'''
fig, ax = plt.subplots(figsize=(7,6))
for i, pos in enumerate(positions):
    x = pos[0]
    y = pos[1]
    ax.scatter(x, y, marker='.')
    ax.text(x, y, '{:d}'.format(i), fontsize=20)
#plt.show()
'''

assert n_res == positions.shape[0]
n_atoms = 4*n_res

# res_map[i] => j gives residue j to which atom i belongs (4 atoms per res)
res_map = []
for i in range(n_res):
    for j in range(4): res_map.append(i)
for i in range(n_res, n_res+n_vsites):
    res_map.append(i)

univ = MDAnalysis.Universe.empty(n_atoms+n_vsites, n_res+n_vsites, atom_resindex=res_map, trajectory=True)
univ.add_TopologyAttr('name')
univ.add_TopologyAttr('resname')
univ.add_TopologyAttr('id')
univ.add_TopologyAttr('resid')
univ.add_TopologyAttr('mass')
univ.add_TopologyAttr('charge')
#univ.add_TopologyAttr('index')

if n_vsites == 0:
    atoms = univ.atoms
    residues = univ.residues
else:
    atoms = univ.atoms[:-n_vsites]
    residues = univ.residues[:-n_vsites]

for i_res, res in enumerate(univ.residues):
    if i_res >= n_res:
        res.atoms[0].name = 'V'
        res.atoms[0].charge = 0
        res.atoms[0].mass = 0
        res.resname = 'V'
        continue

    this_pattern_idx = pattern[i_res]
    res.resname = res_names[this_pattern_idx]

    for j, atm in enumerate(res.atoms):
        if j == 0:
            atm.name = atom_head_names[this_pattern_idx]
        elif j == 1:
            atm.name = 'HO'
        elif j == 2:
            atm.name = 'BOH'
        elif j == 3:
            atm.name = 'HO'
        else:
            raise

        atm.mass = atom_mass[atm.name]
        # Set HO charge to 0 if head group not OH
        if j == 1 and res.atoms[0].name != 'OH':
            atm.charge = 0.0
        else:
            atm.charge = atom_charge[atm.name]

# Top heavies
atoms[::4].positions = positions
# Top hydrogens
atoms[1::4].positions = positions + np.array([0,0,1])
# bottom heavies
#atoms[2::4].positions = positions - np.array([0,0,d])
atoms[2::4].positions = positions + np.array([0.5*d,-(1/6.)*np.sqrt(3)*d,-np.sqrt(2/3.)*d])
# bottom hydrogens
#atoms[3::4].positions = positions - np.array([0,0,d+1])
atoms[3::4].positions = positions + np.array([0.5*d,-(1/6.)*np.sqrt(3)*d,-np.sqrt(2/3.)*d-1])

atoms.ids += args.idx_offset
univ.atoms.write('plate.gro')

tree = cKDTree(atoms.positions)
bond_pairs = sorted(list(tree.query_pairs(r=d*1.01)))
rlist_pairs = sorted(list(tree.query_pairs(r=rcut*1.01)))

# List of pairlist
rlist = {i:[] for i in range(atoms.n_atoms)}
for i,j in rlist_pairs:
    if i == j:
        continue
    rlist[i].append(j)
    rlist[j].append(i)

# list of direct bonds
bond_list = {i:[] for i in range(atoms.n_atoms)}
for i,j in bond_pairs:
    if i == j:
        continue
    bond_list[i].append(j)
    bond_list[j].append(i)

for i in range(atoms.n_atoms):
    bond_list[i] = np.unique(bond_list[i])

# list of 1-3
bond_list_second = {i:[] for i in range(atoms.n_atoms)}
for i in range(atoms.n_atoms):
    second_bond = bond_list[i]
    # Things it's bonded to
    for j in second_bond:
        third_bond = bond_list[j]
        # Things i is 2 steps from
        for k in third_bond:
            if k not in second_bond:
                bond_list_second[i].append(k)

for i in range(atoms.n_atoms):
    bond_list_second[i] = np.unique(bond_list_second[i])

# list of 1-4's
bond_list_third = {i:[] for i in range(atoms.n_atoms)}
for i in range(atoms.n_atoms):
    third_bond = bond_list_second[i]
    # Things its 3 is bonded to
    for j in third_bond:
        fourth_bond = bond_list[j]
        # Things i is 3 steps from
        for k in fourth_bond:
            if k not in bond_list[i] and k not in third_bond:
                bond_list_third[i].append(k)

for i in range(atoms.n_atoms):
    bond_list_third[i] = np.unique(bond_list_third[i])

## Now output topology itp file for this plate
with open('plate.itp', 'w') as fout:
    fout.write('; plate topology generated by make_plate.py script')
    fout.write('\n')

    mtype_str = """
[ moleculetype ]
; Name            nrexcl
Plate             4
"""
    fout.write(mtype_str)
    fout.write('\n')

    fout.write('[ atoms ]\n')
    fout.write(';   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB\n')

    for res in residues:
        fout.write('; residue{:>4d}{:>5s}\n'.format(res.resid, res.resname))

        for atm in res.atoms:
            fout.write('{:>6d}{:>11s}{:>7d}{:>7s}{:>7s}{:>7d}{:>11s}{:>11s}\n'.format(atm.id, atm.name, atm.resid, atm.resname, atm.name, atm.id, str(atm.charge), str(atm.mass)))
            last_id = atm.id
            last_resid = atm.resid
    last_atom_resid = last_resid
    last_atom_id = last_id
    # virtual sites
    vsite_indices = []
    fout.write('; virtual sites - 1-6 are up l, lo l, up r, lo r, up mid, lo mid, 7th is total com\n')
    for i in range(n_vsites):
        last_id += 1
        last_resid += 1
        fout.write('{:>6d}{:>11s}{:>7d}{:>7s}{:>7s}{:>7d}{:>11s}{:>11s}\n'.format(last_id, 'V', last_resid, 'V', 'V', last_id, '0', '0'))
        vsite_indices.append(last_id)

    vsite_indices = vsite_indices[::2][:6]


    fout.write('\n')
    fout.write('[ bonds ]\n')
    fout.write(';  ai    aj funct            c0            c1            c2            c3\n')

    for i,j in bond_pairs:
        if (atoms[i].name == 'HO' or atoms[j].name == 'HO') and np.abs(i-j) != 1:
            continue
        assert atoms[i].id == i+1
        assert atoms[j].id == j+1
        if i < j:
            outstr = '{:>5}{:>6}{:>6}\n'.format(i+1, j+1, 1)
        elif j < i:
            outstr = '{:>5}{:>6}{:>6}\n'.format(j+1, i+1, 1)
        else:
            raise
        fout.write(outstr)

    ## Now v-site bonds
    '''
    for i, (slice_i, slice_j) in enumerate(slice_pairs):
        vsite_idx1 = vsite_indices[slice_i]
        vsite_idx2 = vsite_indices[slice_j]
        outstr = '{:>5}{:>6}{:>6}\n'.format(vsite_idx1, vsite_idx2+1, 6)
        fout.write(outstr)
        outstr = '{:>5}{:>6}{:>6}\n'.format(vsite_idx1+1, vsite_idx2, 6)
        fout.write(outstr)

    for i, j in [(2,3), (4,5)]:
        vsite_idx1 = vsite_indices[-1] + i
        vsite_idx2 = vsite_indices[-1] + j
        outstr = '{:>5}{:>6}{:>6}{:>6.1f}{:>10.1f}\n'.format(vsite_idx1, vsite_idx2, 6, 0.0, 224262.4)
        fout.write(outstr)
    

    fout.write('\n')
    fout.write('[ pairs ]\n')
    fout.write(';  ai    aj funct            c0            c1            c2            c3\n')
    for i, neigh in bond_list_second.items():
        if atoms[i].name == 'HO':
            continue
        for j in neigh:
            if atoms[j].name == 'HO':
                continue
            if j <= i:
                continue
            outstr = '{:>5}{:>6}{:>6}\n'.format(i+1, j+1, 1)
            fout.write(outstr)    
    '''

    fout.write('\n')
    fout.write('[ angles ]\n')
    fout.write(';  ai    aj    ak funct            c0            c1            c2            c3\n')
    for res in residues:
        atoms = res.atoms
        fout.write('{:>6}{:>6}{:>6}{:>6}\n'.format(atoms[0].id, atoms[2].id, atoms[3].id, 1))
        fout.write('{:>6}{:>6}{:>6}{:>6}\n'.format(atoms[2].id, atoms[0].id, atoms[1].id, 1))


    dihed_vsites = [[(0,1,5), (1,5,3)], 
                    [(1,5,3), (5,3,2)],
                    [(5,3,2), (3,2,4)],
                    [(3,2,4), (2,4,0)],
                    [(2,4,0), (4,0,1)],
                    [(4,0,1), (0,1,5)]]

    fout.write('\n')
    if n_vsites > 0:

        fout.write('[ dihedrals ]\n')
        fout.write(';  ai    aj    ak    al funct\n')
        for dihed_indices in dihed_vsites:
            i,j,k = dihed_indices[0]
            j,k,l = dihed_indices[1]

            i = vsite_indices[i]
            j = vsite_indices[j]
            k = vsite_indices[k]
            l = vsite_indices[l]

            fout.write('{:>6}{:>6}{:>6}{:>6}{:>6}\n'.format(i, j, k, l, 2))
            fout.write('{:>6}{:>6}{:>6}{:>6}{:>6}\n'.format(i+1, j+1, k+1, l+1, 2))

        fout.write('\n')
        fout.write('[ virtual_sitesn ]\n')
        fout.write('; Vsite funct     from  \n')
        for i, this_slice in enumerate(slices):
            # Top slice
            vsite_idx = vsite_indices[i]
            
            outstr = '{:>6}{:>4}'.format(vsite_idx, 2)
            for slice_idx in this_slice:
                res = residues[slice_idx]
                atm = res.atoms[0]
                outstr += '{:>6}'.format(atm.id)
            outstr += '\n'
            fout.write(outstr)

            # Bottom slice (BOH)
            vsite_idx = vsite_indices[i] + 1

            outstr = '{:>6}{:>4}'.format(vsite_idx, 2)
            for slice_idx in this_slice:
                res = residues[slice_idx]
                atm = res.atoms[2]
                outstr += '{:>6}'.format(atm.id)
            outstr += '\n'
            fout.write(outstr)

        fout.write('\n')

    outstr = """
; Include Position restraint file
#ifdef POSRES
#include "posre.itp"
#endif
"""
    fout.write(outstr)
    fout.write('\n')

    outstr = """
; Include Position restraint file
#ifdef POSRESXY
#include "posre_xy.itp"
#endif
"""
    fout.write(outstr)


# Write out posre.itp file
with open('posre.itp', 'w') as fout:
    fout.write('; position restraints for plate\n')
    fout.write('\n')
    fout.write('[ position_restraints ]\n')
    fout.write(';  i funct       fcx        fcy        fcz\n')

    for atm in univ.atoms:
        if atm.name != 'HO':
            fout.write('{:>4d}{:>5d}{:>11s}{:>11s}{:>11s}\n'.format(atm.id, 1, '1000', '1000', '1000'))



