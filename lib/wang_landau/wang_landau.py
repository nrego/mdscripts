from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

from matplotlib import pyplot as plt

import matplotlib as mpl

from IPython import embed

import cPickle as pickle
import os


# Generate SAM configurations (from a list of positions and methyl positions)
#   That span over an arbitrary order parameter
class WangLandau:

    @staticmethod
    def is_flat(hist, s=0.7):
        avg = hist.mean()
        abs_diff = np.abs(hist - avg)
        test = abs_diff < avg*s

        return test.all()

    def __init__(self, positions, fn, bins, eps=1e-8, max_iter=60000, f_init=1, f_scale=0.5):
        self.positions = positions
        self.fn = fn
        assert bins.ndim == 1
        self.bins = bins

        self._init_state()

        ## WL paramters ##
        self.eps = eps
        self.max_iter = max_iter

        self.f_init = f_init
        self.f_scale = f_scale

    def _init_state(self):
        # density of states histogram
        self.density = np.zeros(self.bins.size-1)
        # indices of the sampled points
        self.sampled_pt_idx = np.empty(self.bins.size-1, dtype=object)

    @property
    def N(self):
        return self.positions.shape[0]

    @property
    def pos_idx(self):
        return np.arange(self.N)

    def gen_states(self, k, do_brute=False):
        self._init_state()

        if do_brute:
            n_states = int(binom(self.N, k))
            print("generating all {:d} states by hand".format(n_states))
            self._gen_states_brute(k)
        else:
            self._gen_states_wl(k)

        occ = self.density > 0
        print("{} of {} bins occupied for k={}".format(occ.sum(), self.bins.size-1, k))

    ## Private methods ##
    def _gen_states_brute(self, k):
        # Each element is a list of k indices for a unique configuration
        # Shape: (n_states, k)
        combos = np.array(list(combinations(self.pos_idx, k)))
        rand_idx = np.random.permutation(combos.shape[0])
        combos = combos[rand_idx]

        # Indices of the k methyl positions
        for pt_idx in combos:

            order_param = self.fn(self.positions, pt_idx)
            bin_assign = np.digitize(order_param, self.bins) - 1

            try:
                self.density[bin_assign] += 1
            except IndexError:
                print("ERROR: Value of order param ({}) is out of range of bins".format(order_param))
                exit()

            this_arr = self.sampled_pt_idx[bin_assign]
            if this_arr is None:
                self.sampled_pt_idx[bin_assign] = np.array([pt_idx])

            elif this_arr.shape[0] < 10:
                this_arr = np.vstack((this_arr, pt_idx))

                self.sampled_pt_idx[bin_assign] = np.unique(this_arr, axis=0)

        ## Normalize density of states histogram ##
        self.density /= np.diff(self.bins)
        self.density /= np.dot(np.diff(self.bins), self.density)

    ## Wang-Landau ##
    def _gen_states_wl(self, k):

        n_iter = 0

        wl_states = np.zeros_like(self.density)
        wl_hist = np.zeros_like(self.density)
        wl_entropies = np.zeros_like(self.density)

        # Only check center bins for convergence test (flat histogram)
        #   start w/ full binning, move to center of hist if convergence issues
        center_bin_mask = np.ones_like(wl_hist, dtype=bool)

        pt_idx = np.sort( np.random.choice(self.pos_idx, size=k, replace=False) )

        order_param = self.fn(self.positions, pt_idx)
        bin_assign = np.digitize(order_param, self.bins) - 1

        f = self.f_init

        wl_entropies[bin_assign] += f
        wl_hist[bin_assign] += 1
        wl_states[bin_assign] += 1

        M_iter = 0
        print("M: {}".format(M_iter))
        while f > self.eps:
            n_iter += 1

            pt_idx_new = np.sort( self._trial_move(pt_idx) )
            assert np.unique(pt_idx_new).size == k

            order_param_new = self.fn(self.positions, pt_idx_new)
            bin_assign_new = np.digitize(order_param_new, self.bins) - 1

            # Accept trial move
            if np.log(np.random.random()) < (wl_entropies[bin_assign] - wl_entropies[bin_assign_new]):
                pt_idx = pt_idx_new
                order_param = order_param_new
                bin_assign = bin_assign_new

            # Update histogram and density of states
            wl_entropies[bin_assign] += f
            wl_states[bin_assign] += 1
            wl_hist[bin_assign] += 1
            
            this_arr = self.sampled_pt_idx[bin_assign]
            if this_arr is None:
                self.sampled_pt_idx[bin_assign] = np.array([pt_idx])

            elif this_arr.shape[0] < 10:
                this_arr = np.vstack((this_arr, pt_idx))
                self.sampled_pt_idx[bin_assign] = np.unique(this_arr, axis=0)

            # Remove bin with lowest count
            if n_iter == np.round(0.5*self.max_iter):
                center_bin_mask = (wl_hist > 0)

            if self.is_flat(wl_hist[center_bin_mask]) or n_iter > self.max_iter:
                #embed()
                print(" n_iter: {}".format(n_iter))
                center_bin_mask = (wl_states > 0)
                n_iter = 0
                prev_hist = wl_hist.copy()
                wl_hist[:] = 0
                f = self.f_scale * f
                M_iter += 1
                print("M : {}".format(M_iter))

        wl_entropies -= wl_entropies.max()
        self.density = np.exp(wl_entropies)
        self.density /= np.diff(self.bins)
        self.density /= np.dot(np.diff(self.bins), self.density)

    # Generate a new random point R_j, given existing point R_i
    def _trial_move(self, pt_idx):
        change_pts = np.random.random_integers(0, 1, pt_idx.size).astype(bool)
        same_indices = pt_idx[~change_pts]
        avail_indices = np.setdiff1d(self.pos_idx, same_indices)
        new_pt_idx = pt_idx.copy()

        new_pt_idx[change_pts] = np.random.choice(avail_indices, change_pts.sum(), replace=False)

        return new_pt_idx