from __future__ import division, print_function; __metaclass__ = type

import numpy as np
import scipy.special

# Functions and utilities to construct the Nd x Nd metric tensor from WESTPA dataset
#   Will assume dataset auxdata/coords is present

@np.vectorize
def phi_linear(val, cutoff, lb, ub, L=10):
    #Periodicity - ugh!
    if lb == 0:
        if val >= L - cutoff:
            return (val - L)/(2*cutoff) + 0.5    
    elif lb == L/2:
        if val < cutoff:
            return -val/(2*cutoff) + 0.5

    if val < lb - cutoff or val >= ub + cutoff:
        return 0
    elif val >= lb + cutoff and val < ub - cutoff:
        return 1
    else:
        if val < lb + cutoff:
            return (val - lb)/(2*cutoff) + 0.5

        elif val >= ub - cutoff:
            return -(val - ub)/(2*cutoff) + 0.5
        else:
            raise ValueError("val {} not within bounds!".format(val))


@np.vectorize
def phi_prime_linear(val, cutoff, lb, ub, L=10):
    #Periodicity - ugh!
    if lb == 0:
        if val >= L - cutoff:
            return 1/(2*cutoff)   
    elif lb == L/2:
        if val < cutoff:
            return -1/(2*cutoff)

    if val < lb - cutoff or val >= ub + cutoff or (val >= lb + cutoff and val < ub - cutoff):
        return 0
    else:
        if val < lb + cutoff:
            return 1/(2*cutoff)
        elif val >= ub - cutoff:
            return -1/(2*cutoff)
        else:
            raise ValueError("val {} not within bounds!".format(val))


# Class to handle constructing jacobian for an arbitrary number of input coordinates
#   This class assumes an 'Ntwid' type progress coordinate is being used - it assumes equal
#   cell lengths over dimension
#

#   ndim:  dimensionality of Ntwid coordinate
#   cutoff:  cutoff parameter
#   bounds: (coord_dim x ndim*2) matrix (coord_dim is dimensionality of coords, either 3d or 2d)
#            This matrix defines the boundaries in each dimension (such that N_i is zero if all points
#            are more than cutoff distance from any boundary)
#            Bounds are specified Lb, Ub for ith value for coord dim d as: bounds[d, 2*i] = lb,ub,..
#   slope_fn: user-supplied function - gives slope of N_i,j  (i is one of the ndim dimensions of Ntwid, 
#            j is one of coord_dim dimensions of coordinates)
class Jacobian:

    def __init__(self, cutoff, bounds, phi, phi_prime):
        self.bounds = bounds
        assert bounds.ndim == 2
        assert bounds.shape[1] % 2 == 0

        self.ndim = bounds.shape[1] // 2
        assert self.ndim > 0

        self.coord_dim = bounds.shape[0]
        assert self.coord_dim > 0

        self.cutoff = cutoff

        self.jacobian = np.zeros((self.ndim, self.ndim), dtype=np.float64)

        self.phi = phi
        self.phi_prime = phi_prime

        self.phi_mat = np.zeros(())
        self.phi_prime_mat = None
        self.pref_mat = None


    def reset(self):
        self.jacobian[...] = 0.0
        self.phi_mat = None
        self.phi_prime_mat = None

    # x is (N x coord_dim) vector of coordinates
    # Get phi, phi_prime
    def gen_phi(self, x):
        self.phi_mat = np.zeros((x.shape[0], self.ndim), dtype=np.float32)
        self.phi_prime_mat = np.zeros_like(self.phi_mat)
        self.pref_mat = np.zeros_like(self.phi_mat)

        # This can certainly be done better
        for i in range(self.ndim):
            # Ugh!
            # j == 0 x; j == 1 y, optionally j == 3 is z
            for j in range(self.coord_dim):
                lb, ub = self.bounds[j,2*i:2*i+2]
                phi = self.phi(x[j::self.coord_dim], self.cutoff, lb, ub)
                phi_prime = self.phi_prime(x[j::self.coord_dim], self.cutoff, lb, ub)

                self.phi_mat[j::self.coord_dim, i] = phi
                self.phi_prime_mat[j::self.coord_dim, i] = phi_prime

            # Double ugh!
            for j in range(self.coord_dim):
                self.pref_mat[j::self.coord_dim] = 1
                for k in range(self.coord_dim):
                    if k != j:
                        self.pref_mat[j::self.coord_dim] *= self.phi_mat[k::self.coord_dim]

    def compute_jac(self, x):
        self.reset()

        x = self.gen_phi(x)

        self.jacobian = self.pref_mat * self.phi_prime_mat


def produce_plate_bounds():
    xbounds = np.linspace(1.8, 3.2, 5)
    ybounds = np.linspace(1.83, 6.17, 6)
    zbounds = np.linspace(4.122, 7.881, 6)

    lims = np.zeros((3,200))

    for i in range(4):
        for j in range(5):
            for k in range(5):
                xlb = xbounds[i]
                xub = xbounds[i+1]
                ylb = ybounds[j]
                yub = ybounds[j+1]
                zlb = zbounds[k]
                zub = zbounds[k+1]

                n_idx = 2*(i*25 + j*5 + k)

                lims[0, n_idx:n_idx+2] = xlb, xub
                lims[1, n_idx:n_idx+2] = ylb, yub
                lims[2, n_idx:n_idx+2] = zlb, zub

    return lims


# Precalculate a few constants
rcut2 = rcut*rcut;
sig2 = 2.0*sig*sig;

# Normalisation for a gaussian truncated at x = rcut,
# shifted and scaled to be continuous 
normconst = sqrt( M_PI * sig2 ) * erf( rcut / (sqrt(2.0)*sig) )
    - 2*rcut*exp( - rcut2 / sig2 );

# Variables for integrating phi over the probe volume
preerf = sqrt( 0.5 * M_PI * sig * sig ) / normconst;
prelinear = exp( - rcut2 / sig2 ) / normconst;

def phi_actual(val, cutoff, lb, ub, width=0.01):

    if val < lb - cutoff or val >= ub + cutoff:
        return 0
    elif val >= lb + cutoff and val < ub - cutoff:
        return 1


def phi_prime_actual(val, cutoff, lb, ub, width=0.01):
    pass

# Calculates average metric tensor for n_iter
# Coord dim is 2 (x,y) or 3 (x,y,z)
def get_tensor(n_iter, iter_group, coord_dim):
    weights = iter_group['seg_index']['weight']

    try:
        coords = iter_group['auxdata/coords']
    except KeyError:
        raise KeyError('Expecting dataset to have "auxdata/coords" entry for each iteration')

    nsegs = coords.shape[0]
    nsteps = coords.shape[1] - 1

    bounds = np.zeros((2,4))

    bounds[0] = bounds[1] = 0,5, 5,10

    jac = Jacobian(1, bounds, phi_linear, phi_prime_linear)

    avg_tensor = np.zeros((2,2), dtype=np.float64)

    for iseg in range(nsegs):
        i_tensor = np.zeros((2,2), dtype=np.float64)

        for istep in range(nsteps):

            x = coords[iseg, istep]
            jac.compute_jac(x)

            i_tensor += np.dot(jac.jacobian.T, jac.jacobian)

        i_tensor /= nsteps

        avg_tensor += weights[iseg] * i_tensor

    return avg_tensor
    