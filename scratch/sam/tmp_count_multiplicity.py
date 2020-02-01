from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import sympy

import itertools
from scipy.special import binom

from scratch.sam.util import *

from functools import reduce

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

state = State(np.array([], dtype=int), ny=3, nz=3)
max_ko = state.k_o
max_noo = state.n_oo
max_noe = state.n_oe

max_N = 16 
vals_p = vals_q = np.arange(max_N+1)

vals_ko = np.arange(max_ko+1)
vals_noo = np.arange(max_noo+1)
vals_noe = np.arange(max_noe+1)

pp, qq, xx, yy, zz = np.meshgrid(vals_p, vals_q, vals_ko, vals_noo, vals_noe, indexing='ij')

# Count of density of states, binned by p, q, k_o, n_oo, and n_oe
omega = np.zeros_like(pp)

for p, q in itertools.product(vals_p, repeat=2):
    
    N = p*q

    if N == 0 or N > max_N:
        print("\n##############\n[Skipping p={} q={} (N={})]\n################\n".format(p,q,N))
        continue

    indices_to_choose = np.arange(N)

    print("\n\nP = {} Q = {} (N = {})".format(p,q,N))
    print("################\n")

    idx_p = np.digitize(p, vals_p) - 1
    idx_q = np.digitize(q, vals_q) - 1
    
    for k_c in np.arange(N+1):
        n_combos = int(binom(N, k_c))
        print("  kc = {}  (combos: {})".format(k_c, n_combos))

        idx_ko = np.digitize(N-k_c, vals_ko) - 1
        
        for pt_idx in itertools.combinations(indices_to_choose, k_c):
            pt_idx = np.array(pt_idx).astype(int)

            state = State(pt_idx, ny=p, nz=q)

            idx_noo = np.digitize(state.n_oo, vals_noo) - 1
            idx_noe = np.digitize(state.n_oe, vals_noe) - 1

            omega[idx_p, idx_q, idx_ko, idx_noo, idx_noe] += 1

