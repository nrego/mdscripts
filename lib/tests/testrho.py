from __future__ import division, print_function
from rhoutils import phi_1d, rho, gaus_1d
test_fast_phi = True
try:
    from rhoutils import fast_phi
except ImportError:
    test_fast_phi = False
from math import erf

import scipy.integrate

import numpy as np


DTYPE = np.float32

# Tolerance for array equivalence tests, etc
arr_tol = 1e-5

class TestPhi:

    xlim = 20.0
    sigma = 2.4
    cutoff = 7.0
    # Number of neighbor points
    n_pts = 10000

    @staticmethod
    def phi_fn(x, sigma, cutoff):

        two_sig_sq = 2*sigma**2
        phic = np.exp( - cutoff**2 / two_sig_sq )

        norm = np.sqrt(2*np.pi) * sigma * erf( cutoff / np.sqrt(two_sig_sq) ) \
        - 2*cutoff*np.exp( -cutoff**2 / two_sig_sq )

        ret_arr = ( np.exp( -x**2 / two_sig_sq ) - np.exp( -cutoff**2 / two_sig_sq ) ) / norm
        mask = x**2 > cutoff**2
        ret_arr[mask] = 0.0


        return ret_arr

    @staticmethod
    def gaus_fn(x, sigma):

        norm = np.sqrt(2*np.pi) * sigma

        return np.exp(-x**2 / (2*sigma**2)) / norm

    # calculate phivals (from internal function, above) between [-cutoff and cutoff]
    # calculate rhovals for a randomly initialized set of distance vectors
    @classmethod
    def setup_class(cls):
        cls.vals = np.linspace(-cls.xlim, cls.xlim, 1001, dtype=DTYPE)
        cls.phivals = np.zeros_like(cls.vals)
        cls.gausvals = np.zeros_like(cls.vals)

        #for i, xpt in enumerate(cls.vals):
        cls.phivals = cls.phi_fn(cls.vals, cls.sigma, cls.cutoff)
        cls.gausvals = cls.gaus_fn(cls.vals, cls.sigma)

        # Internal consistency check to make sure our phivals integrate to 1
        assert np.abs(np.trapz(cls.phivals, cls.vals) - 1) < 1e-5
        assert np.abs(np.trapz(cls.gausvals, cls.vals) - 1) < 1e-7

        # maximum component in any direction. Calculated by assuming the max cutoff
        #   dist is sqrt(3 * cutoff**2)
        max_comp = np.sqrt(3*cls.cutoff**2)
        cls.dist_vectors = np.random.uniform(low=-max_comp, high=max_comp, size=(cls.n_pts, 3)).astype(DTYPE)

        xpts = cls.dist_vectors[:,0]
        ypts = cls.dist_vectors[:,1]
        zpts = cls.dist_vectors[:,2]

        cls.rhovals = np.zeros((cls.n_pts,), dtype=DTYPE)
        phi_x = cls.phi_fn(xpts, cls.sigma, cls.cutoff)
        phi_y = cls.phi_fn(ypts, cls.sigma, cls.cutoff)
        phi_z = cls.phi_fn(zpts, cls.sigma, cls.cutoff)

        cls.rhovals = phi_x*phi_y*phi_z

    def test_phi_1d_vals(self, arr_tol=1e-7):
        test_phivals = phi_1d(self.vals, sigma=self.sigma, sigma_sq=self.sigma**2, cutoff=self.cutoff, cutoff_sq=self.cutoff**2)

        max_diff = np.abs((test_phivals - self.phivals)).max()
        assert max_diff < arr_tol, "maximum difference ({}) > tolerance ({})".format(max_diff, arr_tol)


    def test_gaus_1d_vals(self, arr_tol=1e-7):
        test_gvals = gaus_1d(self.vals, self.sigma, self.sigma**2)

        max_diff = np.abs((test_gvals - self.gausvals)).max()
        assert max_diff < arr_tol, "maximum difference ({}) > tolerance ({})".format(max_diff, arr_tol)

    def test_phi_1d_normalized(self, arr_tol=1e-6):

        test_phivals = phi_1d(self.vals, sigma=self.sigma, sigma_sq=self.sigma**2, cutoff=self.cutoff, cutoff_sq=self.cutoff**2)

        integ = scipy.integrate.trapz(test_phivals, self.vals)

        assert np.abs(1 - integ) < arr_tol, "does not integrate to 1 within tolerance of {}: got {}".format(arr_tol, integ)

    def test_gaus_1d_normalized(self, arr_tol=1e-6):
        test_gvals = gaus_1d(self.vals, self.sigma, self.sigma**2)

        integ = scipy.integrate.trapz(test_gvals, self.vals)

        assert np.abs(1 - integ) < arr_tol, "does not integrate to 1 within tolerance of {}: got {}".format(arr_tol, integ)


    def test_rhovals(self, arr_tol=1e-6):

        test_rhovals = rho(self.dist_vectors, sigma=self.sigma, sigma_sq=self.sigma**2, cutoff=self.cutoff, cutoff_sq=self.cutoff**2)

        max_diff = np.abs((test_rhovals - self.rhovals).max())

        assert max_diff < arr_tol, "maximum difference ({}) > tolerance ({})".format(max_diff, arr_tol)

        #assert (test_rhovals == 0.0).sum() == (self.rhovals == 0).sum(), "different number of rho=zero values"
        max_zero_diff = test_rhovals[self.rhovals==0.0].max()
        assert max_zero_diff < arr_tol
        max_zero_diff = self.rhovals[test_rhovals==0.0].max()
        assert max_zero_diff < arr_tol
