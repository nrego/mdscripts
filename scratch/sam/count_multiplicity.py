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

from lib.wang_landau import WangLandau

import numpy as np
import sympy

import itertools
from scipy.special import binom

from scratch.sam.util import *

from functools import reduce

# Force WL if number of configs for this P,Q,k_o is greater than this number
MAX_MULT = 2e6


def get_order(pt_idx, m_mask, p, q):
    state = State(pt_idx.astype(int), ny=p, nz=q)

    return (state.n_oo, state.n_oe)


parser = argparse.ArgumentParser("Count density of states in n_oo, n_oe (2D) for a given P, Q, and k_o")
parser.add_argument('-p', default=2, type=int,
                    help='P dimension (default: %(default)s)')
parser.add_argument('-q', default=2, type=int,
                    help='Q dimension (default: %(default)s)')
parser.add_argument('--k-o', default=0, type=int,
                    help='k_o (num hydroxyls);  0 <= k_o <= (p*q) (default: %(default)s)')
parser.add_argument('--do-wl', action='store_true',
                    help='Use WL algorithm to estimate D.O.S. (default is true if p*q>16, false otherwise)')
args = parser.parse_args()

p = args.p
q = args.q
k_o = args.k_o

N = p*q
if N < 0:
    raise ValueError('P*Q cannot be negative!')

if k_o < 0 or k_o > N:
    raise ValueError('Invalid k_o ({})'.format(k_o))

mult_total = int(binom(N, k_o))

do_wl = args.do_wl or mult_total > MAX_MULT

state = State(np.array([], dtype=int), ny=p, nz=q)

bins_noo = np.arange(state.n_oo+1)
bins_noe = np.arange(state.n_oe+1)

bins = [bins_noo, bins_noe]


print('Generating states:')
print('################\n')
print('P: {}   Q: {}  (N: {})  k_o: {}'.format(p, q, N, k_o))
print('  ({} total states); do_wl: {}\n\n'.format(mult_total, do_wl))

wl = WangLandau(state.positions, bins, fn=get_order, fn_kwargs={'p':p, 'q':q})
wl.gen_states(k=k_o, do_brute=(not do_wl))

np.savez_compressed('dos_p_{:02d}_q_{:02d}_ko_{:02d}'.format(p,q,k_o), entropies=wl.entropies, density=wl.density, omega_k=wl.omega_k, p=p, q=q, ko=k_o)
