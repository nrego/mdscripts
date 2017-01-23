import numpy as np
from MDAnalysis import SelectionError
from MDAnalysis.analysis.align import alignto
from utils import pbc, mol_broken, merge_sets

MAX_BOND_LEN = 4.0

from scipy.spatial import cKDTree

from selection_specs import sel_spec_heavies_nowall, sel_spec_nowall

# centers 'mol' (specified by mdanalysis selection string, 'mol_spec') in box
#   Translates all atoms in 'univ' so COM of mol is in the center of box. Original
#     orientations maintained
#
#  Modifies MDAnalysis Universe 'univ' in-place by working on atom groups
def center_mol(univ, mol_spec=sel_spec_nowall):

    mol_group = univ.select_atoms(mol_spec)
    box = univ.dimensions[:3]
    assert np.array_equal(univ.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    box_center = box / 2.0
    broken_mol = mol_broken(mol_group)

    if broken_mol:
        com = get_whole_com(mol_group, box)
    else:
        com = mol_group.center_of_mass()

    shift_vec = box_center - com
    univ.atoms.positions += shift_vec

    pbc(univ)

    return univ


# rotates (an already centered!) system to minimize the rotational RMSD
#   of 'mol' against a reference structure. Applies rotation to all
#   atoms in system
#
#  Modifies MDAnalysis Universe 'univ' in-place
def rotate_mol(ref_univ, univ_other, mol_spec=sel_spec_heavies_nowall):

    alignto(univ_other, ref_univ, select=mol_spec)

    return ref_univ


# Gets subgroups of contiguous atoms form 'atom_group' (i.e. all within MAX_BOND_LEN of each other)
def get_fragmented_groups(atom_group, box):
    tree = cKDTree(atom_group.positions)
    pairs = list( tree.query_pairs(r=MAX_BOND_LEN) )

    # List of tuples of indices of groups of atoms within
    groups = np.unique( tree.query_ball_tree(tree, r=MAX_BOND_LEN) )
    groups = sorted(list(groups))
    groups = [tuple(x) for x in groups]
    groups = merge_sets(groups)

    return groups

def get_whole_com(atom_group, box):

    groups = get_fragmented_groups(atom_group, box)
    # find group closest to origin
    min_sum = float('inf')
    min_group_idx = -1
    for i, group in enumerate(groups):
        index_group = np.array(group)

        atoms = atom_group[index_group]

        if atoms.centroid().sum() < min_sum:
            min_sum = atoms.centroid().sum()
            min_group_idx = i

    for i, group in enumerate(groups):
        if i == min_group_idx:
            continue
        index_group = np.array(group)

        atoms = atom_group[index_group]

        centroid = atoms.centroid()
        shift_arr = np.zeros_like(atoms.positions)
        for idx in range(3):
            if centroid[idx] > np.abs(centroid[idx] - box[idx]):
                shift_arr[:, idx] = box[idx]

        atoms.positions -= shift_arr

    return atom_group.center_of_mass()

