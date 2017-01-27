import numpy as np
#from scipy.interpolate import interp1d, UnivariateSpline

cdef extern from "math.h":
    double erf(double)

cimport numpy as np
cimport cython

f_DTYPE = np.float32
ctypedef np.float32_t f_DTYPE_t

i_DTYPE = np.int
ctypedef np.int_t i_DTYPE_t

cdef double MAX_BOND_LEN=4.0


#Enforce pbc for all atoms in u
#  May break up molecules/residues
#  Not this tool doesn't have any knowledge
#  of molecule definitions (e.g. bonds)
#  and works on an atom-wise basis
def pbc(object u):
    assert np.array_equal(u.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    box = u.dimensions[:3]

    pos = u.atoms.positions
    for i, box_len in enumerate(box):
        pos[pos[:,i] > box_len, i] -= box_len
        pos[pos[:,i] < 0, i] += box_len

    u.atoms.positions = pos

# Make specified 'mol_atoms' (an MDAnalysis.AtomGroup instance) whole
#   by reflecting broken atoms across box boundaries
#   
#   This function attempts to reconstruct the whole molecule by moving all 
#     broken atoms into the box image containing the *most* box atoms
def make_grp_whole_again(object u, object mol_atoms):
    assert np.array_equal(u.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    box = u.dimensions[:3]

    if mol_broken(mol_atoms):
        pass

    return mol_atoms

# Check that 'mol_atoms' (an MDAnalysis.AtomGroup) a broken
#   Requires 'bond' data for 'mol_atoms' - a topology file must 
#   be provided
def mol_broken(object mol_atoms):
    assert mol_atoms.bonds.values() is not None, "No bond data provided for mol_atoms - did you specify a topology?"
        
    return mol_atoms.bonds.values().max() > MAX_BOND_LEN


# NOTE: Only works on broken molecule
#   NOT USED
def merge_sets(list_of_sets):
    sets_to_check = sorted(list_of_sets)
    sets_merged = False

    mysets = []

    while len(sets_to_check):
        curr_set = set(sets_to_check.pop())

        for other_set in sets_to_check:
            other_set = set(other_set)

            if len(curr_set.intersection(other_set)):
                curr_set = curr_set.union(other_set)
                sets_merged = True
                other_set = sorted(tuple(other_set))
                other_set = tuple(other_set)
                sets_to_check.remove(other_set)

        curr_set = tuple(curr_set)
        curr_set = sorted(curr_set)
        curr_set = tuple(curr_set)
        mysets.append(curr_set)

    mysets = sorted(list( set(mysets) ))

    if len(mysets) == 1:
        return mysets

    if not sets_merged:
        return mysets

    else:
        return merge_sets(mysets)

cdef double sq_dist(np.ndarray[f_DTYPE_t, ndim=1] pt1, np.ndarray[f_DTYPE_t, ndim=1] pt2):
    return (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2


# Get the minimum image from point 'pt' for a given atom position 'pos' with a given box vector, 'box'
#
#    Essentially, this reflects 'pos' across the box, and returns the position of the reflection that is closest to 'pt'
cdef np.ndarray shift_vectors = np.array([[-1, -1, -1],[-1, -1,  0],[-1, -1,  1],[-1,  0, -1],[-1,  0,  0], [-1,  0,  1], [-1,  1, -1], [-1,  1,  0], [-1,  1,  1],[ 0, -1, -1],[ 0, -1,  0],[ 0, -1,  1],[ 0,  0, -1],[ 0,  0,  1],[ 0,  1, -1],[ 0,  1,  0],[ 0,  1,  1],[ 1, -1, -1],[ 1, -1,  0],[ 1, -1,  1],[ 1,  0, -1],[ 1,  0, 0],[ 1,  0,  1],[ 1,  1, -1],[ 1,  1,  0],[ 1,  1,  1]], dtype=f_DTYPE)
def get_minimum_image(np.ndarray[f_DTYPE_t, ndim=1] pt, np.ndarray[f_DTYPE_t, ndim=1] pos, np.ndarray[f_DTYPE_t, ndim=1] box_vec):

    cdef double min_dist
    cdef np.ndarray new_pos, shift_vector
    cdef np.ndarray this_shift = np.array([0,0,0], dtype=f_DTYPE)

    min_dist = sq_dist(pt, pos)
    min_pos = pos

    for i in range(26):
        shift_vector = shift_vectors[i]
        for j in range(3):
            this_shift[j] = shift_vector[j] * box_vec[j]

        new_pos = pos + this_shift

        new_dist = sq_dist(pt, new_pos)

        if new_dist < min_dist:
            min_dist = new_dist
            min_pos = new_pos


    return min_pos