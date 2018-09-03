# distutils: library_dirs = /usr/lib

import numpy as np
from numpy import searchsorted
# Use from math.h instead??


cimport numpy as np
cimport cython

f_DTYPE = np.float64
ctypedef np.float64_t f_DTYPE_t

i_DTYPE = np.int
ctypedef np.int_t i_DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def get_neglogpdist(np.ndarray[f_DTYPE_t, ndim=1] all_dat, np.ndarray[f_DTYPE_t, ndim=1] bins, np.ndarray[f_DTYPE_t, ndim=1] logweights):
    cdef np.ndarray[f_DTYPE_t, ndim=1] neglogpdist
    cdef i_DTYPE_t bin_idx

    neglogpdist = np.zeros((bins.size-1), dtype=f_DTYPE)
    bin_assign = np.digitize(all_dat, bins) - 1

    for bin_idx in range(bins.size-1):
        assign = bin_assign == bin_idx
        this_logweights = logweights[assign]
        if this_logweights.size == 0:
            neglogpdist[bin_idx] = float('inf')
            continue
        maxval = this_logweights.max()
        this_logweights -= maxval

        neglogpdist[bin_idx] = -np.log(np.exp(this_logweights).sum()) - maxval

    neglogpdist -= neglogpdist.min()


    return neglogpdist

    