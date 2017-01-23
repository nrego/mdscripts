from __future__ import division, print_function
from boxutils import pbc, mol_broken, get_whole_com, get_fragmented_groups, center_mol, merge_sets

from selection_specs import sel_spec_nowall

import os

import MDAnalysis
from MDAnalysis.tests.datafiles import GRO_CUBIC, GRO, TPR, PDB_xvf, TPR_xvf

import numpy as np

DTYPE = np.float32

root_dir = os.path.dirname( os.path.realpath(__file__) )
ARRAY_TOL = 4 # to 4 decimal places


def test_merge_sets_whole():
    sets_to_check = [(0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 16, 17), (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), (4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21), (4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21), (4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), (4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), (6, 7, 8, 9, 10, 11, 12, 13, 14, 15), (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), (8, 9, 14, 15, 16, 17, 18, 19, 20, 21), (14, 15, 16, 17, 18, 19, 20, 21)]
    expected_merge = [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)]

    res_merge = merge_sets(sets_to_check)

    assert expected_merge == res_merge, "Merge whole set fails"

def test_merge_sets_multiple():
    sets_to_check = [(0, 1), (1,2), (2, 4, 5), (3, 6, 7), (8, 10, 11, 12, 13), (9,), (14, 15), (16, 17, 18, 19, 20, 21)]
    expected_merge = [(0, 1, 2, 4, 5), (3, 6, 7), (8, 10, 11, 12, 13), (9,), (14, 15), (16, 17, 18, 19, 20, 21)]
    
    res_merge = merge_sets(sets_to_check)

    assert expected_merge == res_merge, "Merge mult sets fails"

def test_mol_broken_ids_broken_molecule_small():
    ref_top_filepath = os.path.join(root_dir, 'test_data/top.tpr')
    struct_filepath = os.path.join(root_dir, 'test_data/ala_shifted_broken.gro')

    univ = MDAnalysis.Universe(ref_top_filepath, struct_filepath)

    mol_group = univ.select_atoms(sel_spec_nowall)

    assert mol_broken(mol_group)

def test_mol_broken_ids_whole_molecule_small():
    ref_top_filepath = os.path.join(root_dir, 'test_data/top.tpr')
    struct_filepath = os.path.join(root_dir, 'test_data/ala_shifted_whole.gro')

    univ = MDAnalysis.Universe(ref_top_filepath, struct_filepath)

    mol_group = univ.select_atoms(sel_spec_nowall)

    assert not mol_broken(mol_group)

def test_mol_broken_ids_broken_molecule_large():
    univ = MDAnalysis.Universe(TPR, GRO)

    mol_group = univ.select_atoms(sel_spec_nowall)

    assert mol_broken(mol_group)

def test_mol_broken_ids_whole_molecule_large():

    univ = MDAnalysis.Universe(TPR_xvf, PDB_xvf)

    mol_group = univ.select_atoms(sel_spec_nowall)

    assert not mol_broken(mol_group)

def test_get_fragmented_groups_broken_molecule():
    ref_top_filepath = os.path.join(root_dir, 'test_data/top.tpr')
    struct_filepath = os.path.join(root_dir, 'test_data/ala_shifted_broken.gro')

    univ = MDAnalysis.Universe(ref_top_filepath, struct_filepath)

    mol_group = univ.select_atoms(sel_spec_nowall)

    groups = get_fragmented_groups(mol_group, univ.dimensions[:3])

    assert len(groups) == 6, "expected 6 group but got {}".format(len(groups))
    

def test_get_fragmented_groups_whole_molecule():
    ref_top_filepath = os.path.join(root_dir, 'test_data/top.tpr')
    struct_filepath = os.path.join(root_dir, 'test_data/ala_shifted_whole.gro')

    univ = MDAnalysis.Universe(ref_top_filepath, struct_filepath)

    mol_group = univ.select_atoms(sel_spec_nowall)

    groups = get_fragmented_groups(mol_group, univ.dimensions[:3])

    assert len(groups) == 1, "expected 1 group but got {}".format(len(groups))
    assert len(groups[0]) == mol_group.n_atoms

def test_pbc():
    test_set_whole = os.path.join(root_dir, 'test_data/ala_shifted_whole.gro')
    test_set_pbc = os.path.join(root_dir, 'test_data/ala_shifted_broken.gro')
    univ_whole = MDAnalysis.Universe(test_set_whole)
    pbc(univ_whole)

    univ_pbc = MDAnalysis.Universe(test_set_pbc)

    ref_pos = univ_pbc.atoms.positions
    other_pos = univ_whole.atoms.positions

    np.testing.assert_array_almost_equal(ref_pos, other_pos, decimal=ARRAY_TOL)

def test_get_whole_com():
    test_set_whole = os.path.join(root_dir, 'test_data/ala_shifted_whole.gro')
    test_set_broken = os.path.join(root_dir, 'test_data/ala_shifted_broken.gro')
    univ_whole = MDAnalysis.Universe(test_set_whole)
    univ_broken = MDAnalysis.Universe(test_set_broken)

    com_actual = univ_whole.atoms.center_of_mass()

    atom_group = univ_broken.select_atoms(sel_spec_nowall)
    com_res = get_whole_com(atom_group, univ_broken.dimensions[:3])


    np.testing.assert_array_almost_equal(com_actual, com_res, decimal=2)


def test_centering_whole_mol():
    ref_struct_filepath = os.path.join(root_dir, 'test_data/ala_centered.gro')
    ref_top_filepath = os.path.join(root_dir, 'test_data/top.tpr')
    other_filepath = os.path.join(root_dir, 'test_data/ala_shifted_whole.gro')

    ref_univ = MDAnalysis.Universe(ref_top_filepath, ref_struct_filepath)
    other_univ = MDAnalysis.Universe(ref_top_filepath, other_filepath)

    center_mol(other_univ)

    np.testing.assert_array_almost_equal(ref_univ.atoms.positions, other_univ.atoms.positions, decimal=2)

def test_centering_broken_mol():
    ref_struct_filepath = os.path.join(root_dir, 'test_data/ala_centered.gro')
    ref_top_filepath = os.path.join(root_dir, 'test_data/top.tpr')
    other_filepath = os.path.join(root_dir, 'test_data/ala_shifted_broken.gro')

    ref_univ = MDAnalysis.Universe(ref_top_filepath, ref_struct_filepath)
    other_univ = MDAnalysis.Universe(ref_top_filepath, other_filepath)

    center_mol(other_univ)

    np.testing.assert_array_almost_equal(ref_univ.atoms.positions, other_univ.atoms.positions, decimal=2)

