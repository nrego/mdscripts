from __future__ import division, print_function

from boxutils import center_mol, rotate_mol, pbc

from selection_specs import sel_spec_heavies_nowall, sel_spec_nowall

import mdtraj as md

import numpy as np
import math
import itertools

from MDAnalysis import NoDataError
from MDAnalysis.analysis.rms import rmsd
import MDAnalysis

import os, glob

from nose import with_setup
import logging

logger = logging.getLogger('MDAnalysis.core.AtomGroup')
logger.setLevel(logging.INFO)

# RMSD tolerance for fitting (in A)
RMSD_TOL = 0.05

# Precision for np.testing.assert_array_almost_equal: in A
ARR_PREC = 2
ARR_PREC_TIGHT = 4

# check that these atom positions are correct - all within cutoff of center of box
DIST_CUTOFF = 10
PROT_DIST_CUTOFF = 12.2

NUM_ROT_CONFIGS = 10
# These routines perform high-level tests on 'boxutil' functionality.
#   Namely, they test on actual structure files (subdirs './test_structs')
#
#   These test the ability to c


## Return rotation matrix for counterclockwise rotation about 'axis'
#  by theta degrees
def rotation_matrix(axis, theta):
    theta = (math.pi * theta) / 180.0
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])




# Generate translated test-cases from initial MDAnalysis Universe
def gen_shifted_configs(u, parent_dir, filename):
    assert np.array_equal(u.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    filepath = os.path.join(parent_dir, filename)
    base_filename = filename.split('.')[0]

    box = u.dimensions[:3]
    box_center = box / 2
    # Make sure structure is centered in box
    com = u.select_atoms(sel_spec_nowall).center_of_mass()
    err_msg = 'center of mass of mol ({}) not in center of box ({})'.format(com, box_center)
    np.testing.assert_array_almost_equal(box_center, com, decimal=ARR_PREC, err_msg=err_msg)

    shifts = [-1, 0, 1]
    shift_vectors = np.array([p for p in itertools.product(shifts, repeat=3)])

    i = 0
    for shift_vector in shift_vectors:
        
        coord_shift = shift_vector * box/2

        u.atoms.positions += coord_shift

        # Apply PBC
        pbc(u)       
        newfilename = 'shifted_{}_{:02d}.gro'.format(base_filename, i)
        outpath = os.path.join(parent_dir, newfilename)
        u.atoms.write(outpath)
        u.load_new(filepath)
        i += 1

def gen_rotated_configs(u, parent_dir, filename):
    assert np.array_equal(u.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    filepath = os.path.join(parent_dir, filename)
    base_filename = filename.split('.')[0]

    box = u.dimensions[:3]
    box_center = box / 2
    ## Make sure structure is centered in box
    com = u.select_atoms(sel_spec_nowall).center_of_mass()
    err_msg = 'center of mass of mol ({}) not in center of box ({})'.format(com, box_center)
    np.testing.assert_array_almost_equal(box_center, com, decimal=ARR_PREC, err_msg=err_msg)

    rot_vectors = np.random.random((NUM_ROT_CONFIGS, 3))
    rot_degrees = np.random.random(NUM_ROT_CONFIGS) * 360.0

    for i, rot_vector in enumerate(rot_vectors):

        degree_to_rot = rot_degrees[i]
        
        curr_rot_mat = rotation_matrix(rot_vector, degree_to_rot)

        init_positions = u.atoms.positions - com
        new_positions = np.dot(init_positions, curr_rot_mat) + com

        u.atoms.positions = new_positions
     
        newfilename = 'rotated_{}_{:02d}.gro'.format(base_filename, i)
        outpath = os.path.join(parent_dir, newfilename)
        u.atoms.write(outpath)
        u.load_new(filepath)



### Main Tests - iterate through all structure files 

# a list of tuples: (parent_dir, struct_filename)
root_dir = os.path.dirname( os.path.realpath(__file__) )
#root_dir = "/home/nick/research/mdscripts/lib/tests"

test_structs = [ (x[0], x[2]) for x in os.walk('{}/test_structs/'.format(root_dir)) ][1:]


# Check that all atom positions in two structure files are equal
#   after applying function 'fn'
#
#  Used by test routines, below
def check_centering(ref_filepath, ref_toppath, other_glob):
    u_ref = MDAnalysis.Universe(ref_toppath, ref_filepath)

    ref_com = u_ref.select_atoms(sel_spec_nowall).center_of_mass()

    try:
        ref_close_atoms = u_ref.select_atoms('around {:.2f} ({})'.format(DIST_CUTOFF, sel_spec_nowall))
        ref_pos = ref_close_atoms.positions
    except NoDataError:
        ref_close_atoms = None

    other_paths = glob.glob(other_glob)

    for other_filepath in other_paths:
        u_other = MDAnalysis.Universe(ref_toppath, other_filepath)

        # apply centering or rotating function
        center_mol(u_other)

        other_com = u_other.select_atoms(sel_spec_nowall).center_of_mass()

        err_msg = 'Center of masses of mol not equal within precision'
        np.testing.assert_array_almost_equal(ref_com, other_com, err_msg=err_msg, decimal=2)
        
        np.testing.assert_array_almost_equal(u_ref.select_atoms(sel_spec_nowall).positions, u_other.select_atoms(sel_spec_nowall).positions, decimal=2)

        if ref_close_atoms is not None:
            other_pos = u_other.atoms[ref_close_atoms.indices].positions
            rms = np.sqrt(( np.sum((ref_pos - other_pos)**2, axis=1)).mean())
            assert rms < RMSD_TOL, "RMSD ({}) is greater than tolerance ({}) for atoms within {:.2f} A".format(rms, RMSD_TOL, DIST_CUTOFF)

def check_rotation(ref_filepath, other_glob):
    u_ref = MDAnalysis.Universe(ref_filepath)
    ref_pos = u_ref.atoms.positions

    other_paths = glob.glob(other_glob)

    for other_filepath in other_paths:
        u_other = MDAnalysis.Universe(other_filepath)

        # apply centering or rotating function
        rotate_mol(u_ref, u_other)

        other_pos = u_other.atoms.positions

        rms = np.sqrt(( np.sum((ref_pos - other_pos)**2, axis=1)).mean())
        
        assert rms < RMSD_TOL, "RMSD ({}) is greater than tolerance ({})".format(rms, RMSD_TOL)


def setup_shifted():

    for parent_dir, struct_files in test_structs:
        for struct_file in struct_files:
            u = MDAnalysis.Universe(os.path.join(parent_dir, struct_file))
            gen_shifted_configs(u, parent_dir, struct_file)


def teardown_shifted():

    for parent_dir, struct_files in test_structs:
        to_remove = glob.glob(os.path.join(parent_dir, "shifted_*"))

        for file_toremove in to_remove:
            os.unlink(file_toremove)


@with_setup(setup_shifted, teardown_shifted)
def test_shifted_highlevel():

    for parent_dir, struct_files in test_structs:
        for struct_file in struct_files:
            base_filename = struct_file.split('.')[0]

            other_glob = 'shifted_{}_*'.format(base_filename)

            u_refname = os.path.join(parent_dir, struct_file)
            top_ext = os.path.join('topologies', os.path.basename(parent_dir), '{}.tpr'.format(base_filename))
            u_topname = os.path.join(root_dir, top_ext)
            other_fileglob = os.path.join(parent_dir, other_glob)

            yield check_centering, u_refname, u_topname, other_fileglob


def setup_rotated():

    for parent_dir, struct_files in test_structs:
        for struct_file in struct_files:
            u = MDAnalysis.Universe(os.path.join(parent_dir, struct_file))
            gen_rotated_configs(u, parent_dir, struct_file)


def teardown_rotated():

    for parent_dir, struct_files in test_structs:
        to_remove = glob.glob(os.path.join(parent_dir, "rotated_*"))

        for file_toremove in to_remove:
            os.unlink(file_toremove)


@with_setup(setup_rotated, teardown_rotated)
def test_rotated_highlevel():

    for parent_dir, struct_files in test_structs:
        for struct_file in struct_files:
            base_filename = struct_file.split('.')[0]

            other_glob = 'rotated_{}_*'.format(base_filename)

            u_refname = os.path.join(parent_dir, struct_file)
            other_fileglob = os.path.join(parent_dir, other_glob)

            yield check_rotation, u_refname, other_fileglob


# Check how well we can align actual simulation data: the final configuration
#   of a ubiquitin to its starting structure
def test_align_prot_struct():

    top_file_path = os.path.join(root_dir, 'test_data/ubiq/ubiq.tpr')
    ref_struct_path = os.path.join(root_dir, 'test_data/ubiq/ubiq_reference.gro')
    other_struct_path = os.path.join(root_dir, 'test_data/ubiq/ubiq_other.gro')

    ref_univ = MDAnalysis.Universe(top_file_path, ref_struct_path)
    other_univ = MDAnalysis.Universe(top_file_path, other_struct_path)

    center_mol(ref_univ)
    center_mol(other_univ)

    # Rotationally align according to all protein heavies
    rotate_mol(other_univ, ref_univ, mol_spec=sel_spec_heavies_nowall)

    ref_mol = ref_univ.select_atoms(sel_spec_heavies_nowall)
    other_mol = ref_univ.select_atoms(sel_spec_heavies_nowall)
    rms = np.sqrt(( np.sum((ref_mol.atoms.positions - other_mol.atoms.positions)**2, axis=1)).mean())

    assert rms < 5.0, "RMSD ({}) is greater than {} A".format(rms)

# Test how well we can align (center and rotate) after shifting and rotating a starting structure
def test_align_prot_after_shift():

    top_file_path = os.path.join(root_dir, 'test_data/ubiq/ubiq.tpr')
    ref_struct_path = os.path.join(root_dir, 'test_data/ubiq/ubiq_reference.gro')

    ref_univ = MDAnalysis.Universe(top_file_path, ref_struct_path)
    center_mol(ref_univ)

    box = ref_univ.dimensions[:3]
    assert np.array_equal(ref_univ.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    other_univ = MDAnalysis.Universe(top_file_path, ref_struct_path)
    center_mol(other_univ)

    com = other_univ.select_atoms(sel_spec_nowall).center_of_mass()
    # Rotate the system randomly
    rot_vector = np.random.random((3))
    is_neg = np.random.random(3) > 0.5
    rot_vector[is_neg] = - rot_vector[is_neg]
    degree_to_rot = np.random.random() * 360.0
    curr_rot_mat = rotation_matrix(rot_vector, degree_to_rot)

    # Rotate about protein's COM
    init_positions = other_univ.atoms.positions - com
    new_positions = np.dot(init_positions, curr_rot_mat) + com
    other_univ.atoms.positions = new_positions

    # Shift all positions
    shift_vec = np.random.random((3))
    shift_vec /= np.sqrt( np.sum(shift_vec**2) )
    is_neg = np.random.random(3) > 0.5
    shift_vec[is_neg] = -shift_vec[is_neg]
    # Max shift of 10 A in any direction
    dist_to_shift = np.random.random() * 5.0
    shift_vec *= dist_to_shift
    other_univ.atoms.positions += shift_vec

    # ...and apply pbc
    pbc(other_univ)

    ## No check we can retrieve original structure by centering 
    # and fitting the protein's heavy atoms
    center_mol(other_univ)

    new_com = other_univ.select_atoms(sel_spec_nowall).center_of_mass()
    np.testing.assert_array_almost_equal(com, new_com, decimal=ARR_PREC_TIGHT)

    rotate_mol(ref_univ, other_univ, sel_spec_heavies_nowall)

    ref_mol = ref_univ.select_atoms(sel_spec_nowall)
    other_mol = other_univ.select_atoms(sel_spec_nowall)

    rmsd_mol = rmsd(ref_mol.positions, other_mol.positions)
    assert rmsd_mol < 1e-4

    ## Make sure all waters within a reasonable distance of protein are still good
    ref_close_waters = ref_univ.select_atoms('name OW and around {:.1f} {}'.format(PROT_DIST_CUTOFF, sel_spec_nowall))
    other_close_waters = other_univ.select_atoms('name OW and around {:.1f} {}'.format(PROT_DIST_CUTOFF, sel_spec_nowall))

    assert ref_close_waters.n_atoms > 1000
    assert ref_close_waters.n_atoms == other_close_waters.n_atoms

    rmsd_waters = rmsd(ref_close_waters.positions, other_close_waters.positions)
    assert rmsd_waters < 1e-4
