from __future__ import division, print_function; __metaclass__ = type
import numpy as np

import MDAnalysis

import argparse
from util import *

from scipy.spatial import cKDTree

from IPython import embed

n_mid = 5
d = 2.5

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

n_res = n_mid + 2*fib(n_mid, n_mid)
pattern = np.zeros(n_res, dtype=int)
assert pattern.size == n_res

positions = gen_plate_position(n_mid, d)

assert n_res == positions.shape[0]
n_atoms = 4*n_res

# res_map[i] => j gives residue j to which atom i belongs (4 atoms per res)
res_map = []
for i in range(n_res):
    for j in range(4): res_map.append(i)

univ = MDAnalysis.Universe.empty(n_atoms, n_res, atom_resindex=res_map, trajectory=True)
univ.add_TopologyAttr('name')
univ.add_TopologyAttr('resname')
univ.add_TopologyAttr('id')
univ.add_TopologyAttr('resid')
univ.add_TopologyAttr('mass')
univ.add_TopologyAttr('charge')
#univ.add_TopologyAttr('index')

atoms = univ.atoms
residues = univ.residues
for i_res, res in enumerate(univ.residues):
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

atoms[::4].positions = positions
atoms[1::4].positions = positions + np.array([0,0,1])
atoms[2::4].positions = positions - np.array([0,0,d])
atoms[3::4].positions = positions - np.array([0,0,d+1])

atoms.write('plate.gro')

tree = cKDTree(atoms.positions)
bond_pairs = sorted(list(tree.query_pairs(r=d*1.005)))
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
Plate            2
"""
    fout.write(mtype_str)
    fout.write('\n')

    fout.write('[ atoms ]\n')
    fout.write(';   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB\n')

    for res in residues:
        fout.write('; residue{:>4d}{:>5s}\n'.format(res.resid, res.resname))

        for atm in res.atoms:
            fout.write('{:>6d}{:>11s}{:>7d}{:>7s}{:>7s}{:>7d}{:>11s}{:>11s}\n'.format(atm.id, atm.name, atm.resid, atm.resname, atm.name, atm.id, str(atm.charge), str(atm.mass)))

    fout.write('\n')
    fout.write('[ bonds ]\n')
    fout.write(';  ai    aj funct            c0            c1            c2            c3\n')

    for i,j in bond_pairs:
        if (atoms[i].name == 'HO' or atoms[j].name == 'HO') and np.abs(i-j) != 1:
            continue
        if i < j:
            outstr = '{:>5}{:>6}{:>6}\n'.format(i+1, j+1, 1)
        elif j < i:
            outstr = '{:>5}{:>6}{:>6}\n'.format(j+1, i+1, 1)
        else:
            raise
        fout.write(outstr)

    fout.write('\n')
    fout.write('[ pairs ]\n')
    fout.write(';  ai    aj funct            c0            c1            c2            c3\n')
    for i in range(univ.atoms.n_atoms):
        for j in bond_list_second[i]:
            if univ.atoms[i].name != 'HO' and univ.atoms[j].name != 'HO':
                if i < j:
                    outstr = '{:>5}{:>6}{:>6}\n'.format(i+1, j+1, 1)
                elif j < i:
                    outstr = '{:>5}{:>6}{:>6}\n'.format(j+1, i+1, 1)
                else:
                    continue
                fout.write(outstr)

    fout.write('\n')
    fout.write('[ angles ]\n')
    fout.write(';  ai    aj    ak funct            c0            c1            c2            c3\n')
    for res in residues:
        atoms = res.atoms
        fout.write('{:>6}{:>6}{:>6}{:>6}\n'.format(atoms[0].id, atoms[2].id, atoms[3].id, 1))
        fout.write('{:>6}{:>6}{:>6}{:>6}\n'.format(atoms[2].id, atoms[0].id, atoms[1].id, 1))

    fout.write('\n')