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

#Enforce pbc for all atoms in u
def pbc(object u):
    assert np.array_equal(u.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    box = u.dimensions[:3]

    pos = u.atoms.positions
    for i, box_len in enumerate(box):
        pos[pos[:,i] > box_len, i] -= box_len
        pos[pos[:,i] < 0, i] += box_len

    u.atoms.positions = pos

# Make molecule whole by constructing surrounding boxes
def make_whole(object u, str selection_spec='all prot'):
    assert np.array_equal(u.dimensions[3:], np.ones(3)*90), "Not a cubic box!"

    box = u.dimensions[:3]

    u.select_atoms(selection_spec)
