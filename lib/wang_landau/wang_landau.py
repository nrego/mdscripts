from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

from matplotlib import pyplot as plt

import matplotlib as mpl

from IPython import embed

import sys, os

import time

# Generate SAM configurations (from a list of positions and methyl positions)
#   That span over an arbitrary order parameter
class WangLandau:

    @staticmethod
    def is_flat(hist, s=0.7):
        if (hist > 0).sum() < 2:
            return False

        avg = hist.mean()
        abs_diff = np.abs(hist - avg)
        test = abs_diff < avg*(1-s)

        return test.all()

    def __init__(self, positions, bins, fn, fn_kwargs=None, eps=1e-8, max_iter=60000, f_init=1, f_scale=0.5):
        self.fn = fn
        self.fn_kwargs = fn_kwargs if fn_kwargs is not None else {}

        self.positions = positions

        # List of the bin boundaries for each dimension of the progress coord
        self.bins = list(bins)

        for i, this_bin in enumerate(self.bins):
            this_bin = np.array(this_bin)
            assert this_bin.ndim == 1
            self.bins[i] = this_bin
        
        self._init_state()

        ## WL paramters ##
        self.eps = eps
        self.max_iter = max_iter

        self.f_init = f_init
        self.f_scale = f_scale

        self._k_current = None

    def _init_state(self):
        # density of states histogram
        self.density = np.zeros(self.shape)
        self.entropies = np.zeros(self.shape)
        # indices of the sampled points - a record of patterns in each bin
        self.sampled_pt_idx = np.empty(self.shape, dtype=object)
        self._k_current = None

    # This is different than the dimensionality of self.bins, which is a
    #    1d array of (possibly multiple) bin boundaries.
    #    The number of bin boundaries is what we're calling ndim here -
    #    i.e., the dimensionality of whatever order parameter we're trying
    #    to determine the multiplicity
    @property
    def n_dim(self):
        return len(self.bins)

    @property
    def shape(self):
        return tuple(b.size for b in self.bins)

    @property
    def n_bins(self):
        return np.prod([b.size for b in self.bins])

    @property
    def N(self):
        return self.positions.shape[0]

    @property
    def k_current(self):
        return self._k_current

    @property
    def omega(self):
        try:
            return int(binom(self.N, self._k_current)) * self.density / self.density.sum()
        except TypeError:
            raise ValueError("k not yet set: use gen_states")

    @property
    def omega_k(self):
        try:
            return int(binom(self.N, self._k_current))
        except TypeError:
            raise ValueError("k not yet set: use gen_states")

    @property
    def pos_idx(self):
        return np.arange(self.N)

    # Run WL for given fixed k (# methyls)
    def gen_states(self, k, do_brute=False, hist_flat_tol=0.7):
        self._init_state()

        self._k_current = k

        try:
            assert hist_flat_tol > 0.0 and hist_flat_tol <= 1.0
        except AssertionError:
            raise ValueError('parameter hist_flat_tol must be a float between 0.0 and 1.0')

        start_time = time.time()
        if do_brute:
            n_states = int(binom(self.N, k))
            print("generating all {:d} states by hand".format(n_states))
            self._gen_states_brute(k)
        else:
            self._gen_states_wl(k, hist_flat_tol)

        end_time = time.time()

        occ = self.density > 0
        print("{} of {} bins occupied for k={}".format(occ.sum(), self.n_bins, k))
        print("time: {:1.1f} s".format(end_time-start_time))

    ## Private methods ##
    def _gen_states_brute(self, k):
        # Each element is a list of k indices for a unique configuration
        # Shape: (n_states, k)
        combos = np.array(list(combinations(self.pos_idx, k)))
        np.random.seed()
        rand_idx = np.random.permutation(combos.shape[0])
        combos = combos[rand_idx]
        
        # Indices of the k methyl positions
        for pt_idx in combos:

            m_mask = np.zeros(self.N, dtype=bool)
            try:
                m_mask[pt_idx] = True
            except IndexError: # all hydroxyl
                pass

            order_param = self.fn(pt_idx, m_mask, **self.fn_kwargs)
            try:
                bin_assign = tuple(np.digitize(op, b) - 1 for op, b in zip(order_param, self.bins))
            except TypeError:
                print("ERROR: Order param not iterable (is your function returning an ndim length iterable?)")
            
            try:
                self.density[bin_assign] += 1
            except IndexError:
                print("ERROR: Value of order param ({}) is out of range of bins".format(order_param))
                sys.exit()

            # Add this point to its spot in the binning
            this_arr = self.sampled_pt_idx[bin_assign]
            if this_arr is None:
                self.sampled_pt_idx[bin_assign] = np.array([pt_idx])
            elif this_arr.shape[0] < 10:
                this_arr = np.vstack((this_arr, pt_idx))
                self.sampled_pt_idx[bin_assign] = np.unique(this_arr, axis=0)

        ## Normalize density of states histogram ##
        np.seterr(divide='ignore')
        self.entropies = np.log(self.density)
        self.entropies -= np.ma.masked_invalid(self.entropies).max()

    ## Wang-Landau ##
    def _gen_states_wl(self, k, hist_flat_tol):

        n_iter = 0
        
        wl_states = np.zeros_like(self.density)
        wl_hist = np.zeros_like(self.density)
        wl_entropies = np.zeros_like(self.density)

        # Only check center bins for convergence test (flat histogram)
        #   start w/ full binning, move to center of hist if convergence issues
        center_bin_mask = np.ones_like(wl_hist, dtype=bool)
        iter_update_bin_mask = self.max_iter // 10
        initial_iter_update = self.max_iter // 100

        ## Choose random initial configuration
        pt_idx = np.sort( np.random.choice(self.pos_idx, size=k, replace=False) )
        m_mask = np.zeros(self.N, dtype=bool)
        m_mask[pt_idx] = True

        order_param = self.fn(pt_idx, m_mask, **self.fn_kwargs)
        try:
            bin_assign = tuple(np.digitize(op, b) - 1 for op, b in zip(order_param, self.bins))
        except TypeError:
            print("ERROR: Order param not iterable (is your function returning an ndim length iterable?)")

        f = self.f_init

        wl_entropies[bin_assign] += f
        wl_hist[bin_assign] += 1
        wl_states[bin_assign] += 1

        M_iter = 0
        print("M: {}".format(M_iter))

        while f > self.eps:
            
            n_iter += 1

            pt_idx_new = self._trial_move(pt_idx)
            m_mask = np.zeros(self.N, dtype=bool)
            m_mask[pt_idx_new] = True
            assert np.unique(pt_idx_new).size == k
            
            # Outside call is slow!
            order_param_new = self.fn(pt_idx_new, m_mask, **self.fn_kwargs)
            try:
                bin_assign_new = tuple(np.digitize(op, b) - 1 for op, b in zip(order_param_new, self.bins))
            except TypeError:
                print("ERROR: Order param not iterable (is your function returning an ndim length iterable?)")

            # Accept trial move
            if np.log(np.random.random()) < (wl_entropies[bin_assign] - wl_entropies[bin_assign_new]):
                pt_idx = pt_idx_new
                order_param = order_param_new
                bin_assign = bin_assign_new

            # Update histogram and density of states
            try:
                wl_entropies[bin_assign] += f
                wl_states[bin_assign] += 1
                wl_hist[bin_assign] += 1
            except:
                embed()
            
            ## Store configurations that fall into this bin
            this_arr = self.sampled_pt_idx[bin_assign]
            if this_arr is None:
                self.sampled_pt_idx[bin_assign] = np.array([pt_idx])

            elif this_arr.shape[0] < 10:
                this_arr = np.vstack((this_arr, pt_idx))
                self.sampled_pt_idx[bin_assign] = np.unique(this_arr, axis=0)

            # Periodically check to ignore bins with zero count
            if ((n_iter + 1) % iter_update_bin_mask == 0): #or (M_iter == 0 and n_iter > initial_iter_update):
                print("    iter: {} updating bin mask".format(n_iter+1))
                old_center_bin_mask = center_bin_mask.copy()
                if M_iter > 0:
                    center_bin_mask = (wl_hist > 0) | old_center_bin_mask
                else:
                    center_bin_mask = (wl_hist > 0)
                print("      (from {} bins to {} bins)".format(old_center_bin_mask.sum(), center_bin_mask.sum()))

            is_flat = self.is_flat(wl_hist[center_bin_mask], hist_flat_tol)

            # Bin's flat - move onto next WL iter (i.e. M_iter += 1)
            if  is_flat or n_iter > self.max_iter:
                
                print(" n_iter: {}".format(n_iter+1))
                center_bin_mask = (wl_states > 0) | center_bin_mask
                n_iter = 0
                prev_hist = wl_hist.copy()
                wl_hist[:] = 0
                f = self.f_scale * f
                M_iter += 1
                print("M : {}".format(M_iter))

        occ = wl_entropies > 0
        wl_entropies -= wl_entropies.max()
        self.entropies = wl_entropies.copy()
        self.entropies -= np.ma.masked_invalid(self.entropies).max()
        self.density = np.exp(wl_entropies)
        self.density[~occ] = 0.0

    # Generate a new random point R_j, given existing point R_i

    def _trial_move(self, pt_idx):
    
        avail_indices = np.setdiff1d(self.pos_idx, pt_idx)
        new_idx = np.random.choice(avail_indices)
        change_idx = np.random.randint(0, self._k_current)
    
        new_pt_idx = pt_idx.copy()
        new_pt_idx[change_idx] = new_idx
    
        return new_pt_idx

