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
    def is_flat(hist, s=0.8):
        avg = hist.mean()

        test = hist > avg*s

        return test.all()

    # Generate a new random point R_j, given existing point R_i
    @staticmethod
    def trial_move(pt_idx, k):
        move_idx = np.round(np.random.normal(size=k)).astype(int)
        pt_idx_new = pt_idx + move_idx

        # Reflect new point
        pt_idx_new[pt_idx_new < 0] += 36
        pt_idx_new[pt_idx_new > 35] -= 36

        return pt_idx_new

    def __init__(self, positions, fn, bins):
        self.positions = positions
        self.fn = fn
        assert bins.ndim == 1
        self.bins = bins

        self.density = np.zeros(self.bins.size-1)

    @property
    def N(self):
        return self.positions.shape[0]

    @property
    def pos_idx(self):
        return np.arange(self.N)

    def gen_states(self, k, do_brute=False):

        if do_brute:
            self._gen_states_brute(k)
        else:
            self._gen_states_wl(k)

    ## Private methods
    def _gen_states_brute(k):
        pass


