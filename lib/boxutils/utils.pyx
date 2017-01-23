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
def pbc(object u):
    assert np.array_equal(u.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    box = u.dimensions[:3]

    pos = u.atoms.positions
    for i, box_len in enumerate(box):
        pos[pos[:,i] > box_len, i] -= box_len
        pos[pos[:,i] < 0, i] += box_len

    u.atoms.positions = pos

def mol_broken(object mol_atoms):
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

    if len(mysets) == 1:
        return sorted(mysets)

    mysets = list( np.unique(np.array(mysets)) )

    if not sets_merged:
        return mysets

    else:
        return merge_sets(mysets)