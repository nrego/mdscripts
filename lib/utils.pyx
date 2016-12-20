# distutils: library_dirs = /usr/lib

import numpy
import math
from scipy.spatial import cKDTree

cimport numpy

DTYPE = numpy.float32
ctypedef numpy.float32_t DTYPE_t

# Caluclates value of coarse-grained gaussian for point at r
#   sigma and cutoff in A. only works around this range - have to see
#   About dynamically adjusting the prefactor
#   r is an array of vectors from atom position to grid point position (i.e. grid - atom)
def phi(numpy.ndarray r_array, double sigma, double sigma_sq, double cutoff, double cutoff_sq):

    cdef double x_sq
    cdef double y_sq
    cdef double z_sq
    cdef double phic
    cdef double pref
    cdef double phi_term
    cdef numpy.ndarray r
    cdef int i
    cdef int r_array_shape

    r_array_shape = r_array.shape[0]

    cdef numpy.ndarray[numpy.float32_t, ndim=1] phi_vec
    phi_vec = numpy.zeros(r_array_shape, dtype=DTYPE)

    phic = math.exp(-0.5*(cutoff_sq/sigma_sq))

    pref = (1 / ( (2*math.pi)**(0.5) * sigma * math.erf(cutoff / (2**0.5 * sigma)) - 2*cutoff*phic ))**3

    for i in xrange(r_array_shape):

        r = r_array[i]

        x_sq = r[0]**2
        y_sq = r[1]**2
        z_sq = r[2]**2

        if ((x_sq+y_sq+z_sq) >= cutoff_sq):
            phi_vec[i] = 0.0

        else:

            phi_term = ( (math.exp(-0.5*(x_sq/sigma_sq)) - phic) * (math.exp(-0.5*(y_sq/sigma_sq) - phic)) *
                         (math.exp(-0.5*(z_sq/sigma_sq)) - phic) )
            #print "val: {}".format(pref*phi_term)
            phi_vec[i] = pref * phi_term

    return phi_vec

def _calc_rho(lb, ub, prot_heavies, water_ow, cutoff, sigma, gridpts, npts, rho_prot_bulk, rho_water_bulk):    
    cutoff_sq = cutoff**2
    sigma_sq = sigma**2
    block = ub - lb
    rho_prot_slice = numpy.zeros((block, npts), dtype=numpy.float32)
    rho_water_slice = numpy.zeros((block, npts), dtype=numpy.float32)
    rho_slice = numpy.zeros((block, npts), dtype=numpy.float32)

    # KD tree for nearest neighbor search
    tree = cKDTree(gridpts)

    # i is frame
    for i in xrange(block):

        # position of each atom at frame i
        for pos in prot_heavies[i]:

            #pos = atom.position
            # Indices of all gridpoints within cutoff of atom's position
            neighboridx = numpy.array(tree.query_ball_point(pos, cutoff))
            if neighboridx.size == 0:
                continue
            neighborpts = gridpts[neighboridx]

            dist_vectors = neighborpts[:, ...] - pos

            # Distance array between atom and neighbor grid points
            #distarr = scipy.spatial.distance.cdist(pos.reshape(1,3), neighborpts,
            #                                       'sqeuclidean').reshape(neighboridx.shape)

            phivals = phi(dist_vectors, sigma, sigma_sq, cutoff, cutoff_sq)

            rho_prot_slice[i, neighboridx] += phivals

        for pos in water_ow[i]:
            neighboridx = numpy.array(tree.query_ball_point(pos, cutoff))
            if neighboridx.size == 0:
                continue
            neighborpts = gridpts[neighboridx]

            dist_vectors = neighborpts[:, ...] - pos
            # Distance array between atom and neighbor grid points
            # distarr = scipy.spatial.distance.cdist(pos.reshape(1,3),
            #       neighborpts,'sqeuclidean').reshape(neighboridx.shape)

            phivals = phi(dist_vectors, sigma, sigma_sq, cutoff, cutoff_sq)

            rho_water_slice[i, neighboridx] += phivals

        # Can probably move this out of here and perform at end
        rho_slice[i, :] = rho_prot_slice[i, :]/rho_prot_bulk \
            + rho_water_slice[i, :]/rho_water_bulk

    return (rho_slice, lb, ub)
