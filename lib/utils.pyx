# distutils: library_dirs = /usr/lib

import numpy as np
from math import pi, exp, erf
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d, UnivariateSpline

cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef int min_lup = -15
cdef np.ndarray xvals = np.linspace(min_lup, 0, 10000, dtype=DTYPE)
cdef np.ndarray exp_table = np.exp(xvals, dtype=DTYPE)
exp_lut = interp1d(xvals, exp_table, kind='linear')
#exp_lut = UnivariateSpline(xvals, exp_table, k=1)

# Gets phi for a 1d array of values for the given sigma and cutoff
#
def phi_1d(np.ndarray[DTYPE_t, ndim=1] r_array, double sigma, double sigma_sq, double cutoff, double cutoff_sq):
    cdef np.ndarray[DTYPE_t, ndim=1] r_sq
    cdef double phic
    cdef double pref
    cdef double phi_term
    cdef np.ndarray[DTYPE_t, ndim=1] phi_vec

    r_sq = (r_array**2)

    phic = exp(-(cutoff_sq/(2*sigma_sq)))
    pref = (1 / ( (2*pi)**(0.5) * sigma * erf(cutoff / (2**0.5 * sigma)) - 2*cutoff*phic ))

    phi_vec = pref * (exp_lut(-(r_sq/(2*sigma_sq))) - phic).astype(DTYPE)
    phi_vec[r_sq > cutoff_sq] = 0.0

    return phi_vec
    

# Calculates value of coarse-grained gaussian for each point in r_array (shape (n_pts, 3))
#   sigma and cutoff in A. only works around this range - have to see
def rho(np.ndarray[DTYPE_t, ndim=2] r_array, double sigma, double sigma_sq, double cutoff, double cutoff_sq):

    cdef np.ndarray[DTYPE_t, ndim=1] xpts, ypts, zpts, phi_x, phi_y, phi_z
    cdef int r_array_shape = r_array.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] phi_vec


    xpts = r_array[:,0]
    ypts = r_array[:,1]
    zpts = r_array[:,2]

    phi_x = phi_1d(xpts, sigma, sigma_sq, cutoff, cutoff_sq)
    phi_y = phi_1d(ypts, sigma, sigma_sq, cutoff, cutoff_sq)
    phi_z = phi_1d(zpts, sigma, sigma_sq, cutoff, cutoff_sq)

    phi_vec = phi_x * phi_y * phi_z

    return phi_vec


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out