from __future__ import division, print_function

from boxutils import center_mol, rotate_mol, pbc

from selection_specs import sel_spec_heavies_nowall

import mdtraj as md

import numpy as np
import math
import itertools

import MDAnalysis

import os, glob

from nose import with_setup
import logging

logger = logging.getLogger('MDAnalysis.core.AtomGroup')
logger.setLevel(logging.INFO)

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
    shifts = [-1, 0, 1]
    shift_vectors = np.array([p for p in itertools.product(shifts, repeat=3)])

    i = 0
    for shift_vector in shift_vectors:
        
        coord_shift = shift_vector * box.astype(float)/2

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

    # call to copy is probably unnecessary, as center_of_mass method always
    #    returns a fresh instance of the COM vector (I think)
    #    but good for clarity and just in case
    com = u.select_atoms(sel_spec_heavies_nowall).center_of_mass().copy()

    filepath = os.path.join(parent_dir, filename)
    base_filename = filename.split('.')[0]

    box = u.dimensions[:3]
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
#root_dir = os.path.dirname( os.path.realpath(__file__) )
root_dir = "/home/nick/research/mdscripts/lib/tests"
test_structs = [ (x[0], x[2]) for x in os.walk('{}/test_structs/'.format(root_dir)) ][1:]


# Check that all atom positions in two structure files are equal
#   after applying function 'fn'
#
#  Used by test routines, below
def check_centering(ref_filepath, other_glob):
    u_ref = MDAnalysis.Universe(ref_filepath)
    ref_pos = u_ref.atoms.positions

    other_paths = glob.glob(other_glob)

    for other_filepath in other_paths:
        u_other = MDAnalysis.Universe(other_filepath)

        # apply centering or rotating function
        center_mol(u_other)

        other_pos = u_other.atoms.positions

        assert np.array_equal(ref_pos, other_pos), "RMDS: {}".format( np.sqrt(((ref_pos - other_pos)**2).mean()) )

def check_rotation(ref_filepath, other_glob):
    u_ref = MDAnalysis.Universe(ref_filepath)
    ref_pos = u_ref.atoms.positions

    other_paths = glob.glob(other_glob)

    for other_filepath in other_paths:
        u_other = MDAnalysis.Universe(other_filepath)

        # apply centering or rotating function
        rotate_mol(u_ref, u_other)

        other_pos = u_other.atoms.positions

        assert np.allclose(ref_pos, other_pos, atol=0.02), "RMDS: {}".format( np.sqrt(((ref_pos - other_pos)**2).mean()) )


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
            other_fileglob = os.path.join(parent_dir, other_glob)

            yield check_centering, u_refname, other_fileglob


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


