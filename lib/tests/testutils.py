from __future__ import division, print_function
from utils import phi_1d, rho
from math import exp, erf, sqrt, pi

import scipy.integrate

import numpy as np


DTYPE = np.float32

# Tolerance for array equivalence tests, etc
arr_tol = 1e-7

class TestPhi:

    xlim = 10.0
    sigma = 2.4
    cutoff = 7.0
    # Number of neighbor points
    n_pts = 10000

    @staticmethod
    def phi_fn(x, sigma, cutoff):

        two_sig_sq = 2*sigma**2
        phic = exp( - cutoff**2 / two_sig_sq )

        norm = sqrt(2*pi) * sigma * erf( cutoff / sqrt(two_sig_sq) ) \
        - 2*cutoff*exp( -cutoff**2 / two_sig_sq )

        if x**2 > cutoff**2:
            return 0.0
        else:
            return ( exp( -x**2 / two_sig_sq ) - exp( -cutoff**2 / two_sig_sq ) ) / norm


    # calculate phivals (from internal function, above) between [-cutoff and cutoff]
    # calculate rhovals for a randomly initialized set of distance vectors
    @classmethod
    def setup_class(cls):
        cls.vals = np.linspace(-cls.xlim, cls.xlim, 1000, dtype=DTYPE)
        cls.phivals = np.zeros_like(cls.vals)

        for i, xpt in enumerate(cls.vals):
            cls.phivals[i] = cls.phi_fn(xpt, cls.sigma, cls.cutoff)

        # Internal consistency check to make sure our phivals integrate to 1
        assert np.abs(scipy.integrate.trapz(cls.phivals, cls.vals) - 1) < 1e-7

        # maximum component in any direction. Calculated by assuming the max cutoff
        #   dist is sqrt(3 * cutoff**2)
        max_comp = sqrt(3*cls.cutoff**2)
        cls.dist_vectors = np.random.uniform(low=-max_comp, high=max_comp, size=(cls.n_pts, 3)).astype(DTYPE)

        xpts = cls.dist_vectors[:,0]
        ypts = cls.dist_vectors[:,1]
        zpts = cls.dist_vectors[:,2]

        cls.rhovals = np.zeros((cls.n_pts,), dtype=DTYPE)
        phi_x = np.zeros_like(xpts)
        phi_y = np.zeros_like(ypts)
        phi_z = np.zeros_like(zpts)
        for i in range(cls.n_pts):
            phi_x[i] = cls.phi_fn(xpts[i], cls.sigma, cls.cutoff)
            phi_y[i] = cls.phi_fn(ypts[i], cls.sigma, cls.cutoff)
            phi_z[i] = cls.phi_fn(zpts[i], cls.sigma, cls.cutoff)

        cls.rhovals = phi_x*phi_y*phi_z

    def test_phi_1d_vals(self, arr_tol=1e-7):
        test_phivals = phi_1d(self.vals, sigma=self.sigma, sigma_sq=self.sigma**2, cutoff=self.cutoff, cutoff_sq=self.cutoff**2)

        max_diff = np.abs((test_phivals - self.phivals).max())
        assert max_diff < arr_tol, "maximum difference ({}) > tolerance ({})".format(max_diff, arr_tol)

    def test_phi_1d_normalized(self, arr_tol=1e-6):

        test_phivals = phi_1d(self.vals, sigma=self.sigma, sigma_sq=self.sigma**2, cutoff=self.cutoff, cutoff_sq=self.cutoff**2)

        integ = scipy.integrate.trapz(test_phivals, self.vals)

        assert np.abs(1 - integ) < arr_tol, "does not integrate to 1 within tolerance of {}: got {}".format(arr_tol, integ)

    def test_rhovals(self, arr_tol=1e-4):

        test_rhovals = rho(self.dist_vectors, sigma=self.sigma, sigma_sq=self.sigma**2, cutoff=self.cutoff, cutoff_sq=self.cutoff**2)

        max_diff = np.abs((test_rhovals - self.rhovals).max())

        assert max_diff < arr_tol, "maximum difference ({}) > tolerance ({})".format(max_diff, arr_tol)

        assert (test_rhovals == 0.0).sum() == (self.rhovals == 0).sum(), "different number of rho=zero values"

    ## TODO: testcartesian