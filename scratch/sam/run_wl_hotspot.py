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
from scratch.sam.enumerate_design import GetDelta

from functools import reduce

# Force WL if number of configs for this P,Q,k_o is greater than this number
MAX_MULT = 1e6
MAX_N = 81



# Returns delta_f (w.r.t. pure non-polar), avg f_up, avg f_down
def get_order(pt_idx, m_mask, p, q, delta, do_tot=False):
    

    state = State(pt_idx.astype(int), p=p, q=q)

    # Indices of *polar* groups, candidates for down hotspots
    down_indices = state.avail_indices
    # Indices of *non-polar* groups, candidates for up hotspots
    up_indices = pt_idx

    if down_indices.size == 0:
        delta_f_down = 0
    else:
        new_states, new_energies = delta(m_mask, down_indices)
        delta_f_down = new_energies.mean() 

    if up_indices.size == 0:
        delta_f_up = 0
    else:
        new_states, new_energies = delta(m_mask, up_indices)
        delta_f_up = new_energies.mean() 

    assert delta_f_up >= 0
    assert delta_f_down <= 0

    if do_tot:
        return [(delta_f_up*up_indices.size + delta_f_down*down_indices.size) / (state.N)]
    else:
        return (delta_f_up, delta_f_down)


#### For given size, ko, run WL on pattern 'hotspots'/susceptibility to point mutations
####      'up' hotspots: Average increase in f if a single methyl is switched to hydroxyl
####      'down' hotspots: vice versa
parser = argparse.ArgumentParser("Count density of states in pattern up/down hotspots")
parser.add_argument('-p', default=6, type=int,
                    help='P dimension (default: %(default)s)')
parser.add_argument('-q', default=6, type=int,
                    help='Q dimension (default: %(default)s)')
parser.add_argument('--k-c', default=0, type=int,
                    help='k_o (num hydroxyls);  0 <= k_o <= (p*q) (default: %(default)s)')
parser.add_argument('--do-wl', action='store_true',
                    help='Use WL algorithm to estimate D.O.S. (default is true if p*q>16, false otherwise)')
parser.add_argument('--reg-file', default='sam_data/data/sam_reg_m3.npy', type=str)
parser.add_argument('--reg-file-meth', default='sam_data/data/sam_reg_m3_meth.npy', type=str)
parser.add_argument('-de', default=0.5, type=float,
                    help='Energy bin spacing')
parser.add_argument('--do-expt-tot', action='store_true',
                    help='If true, do WL on expt (average) Delta f (1D WL)')
args = parser.parse_args()

p = args.p
q = args.q
k_c = args.k_c

N = p*q
if N < 0:
    raise ValueError('P*Q cannot be negative!')
if N > MAX_N:
    print('N ({}) > MAX_N (MAX_N), exiting.'.format(N, MAX_N))
    exit()

if k_c < 0 or k_c > N:
    raise ValueError('Invalid k_c ({})'.format(k_c))

## Total number of patterns for this (p,q,ko)
mult_total = int(binom(N, k_c))

# Automatically force WL if total number of patterns is large
do_wl = args.do_wl or mult_total > MAX_MULT

## Pure polar state
state_po = State(np.array([], dtype=int), p=p, q=q)
adj_mat = state_po.adj_mat

# Fit on k_o, n_oo, n_oe
reg = np.load(args.reg_file).item()
alpha_k_o, alpha_n_oo, alpha_n_oe = reg.coef_

# Fit on k_c, n_cc, n_ce
reg_meth = np.load(args.reg_file_meth).item()
alpha_k_c, alpha_n_cc, alpha_n_ce = reg_meth.coef_

## Sanity.
assert np.allclose(alpha_n_oo, alpha_n_cc)
assert np.allclose(alpha_n_oo - alpha_n_oe, alpha_n_ce)

delta = GetDelta(state_po.adj_mat, state_po.ext_count, alpha_k_c, alpha_n_cc, alpha_n_ce)
# Difference in energy between pure polar and pure non-polar patches of this size = 
max_delta_f = alpha_k_o * state_po.k_o + alpha_n_oo * state_po.n_oo + alpha_n_oe * state_po.n_oe

# Non-polar to polar switch
de = args.de

bins_up = np.arange(0, np.ceil(alpha_k_o)+de, de)
bins_down = - bins_up
# Fe of pattern w.r.t pure non-polar (delta f)
bins_delta_f = bins_up.copy()

bins_tot = np.unique(np.append(bins_down, bins_up))

if args.do_expt_tot:
    bins = [bins_tot]
else:
    bins = [bins_up, bins_down]

# DEBUG
#state = State(np.array([ 5,  9, 20, 28, 30, 31, 32, 35]))

if __name__ == '__main__':

    print('Generating states:')
    print('################\n')
    print('P: {}   Q: {}  (N: {})  k_c: {}'.format(p, q, N, k_c))
    print('  ({} total states); do_wl: {}\n\n'.format(mult_total, do_wl))

    kwargs = {
        'p': p,
        'q': q,
        'delta': delta,
        'do_tot': args.do_expt_tot
    }

    wl = WangLandau(state_po.positions, bins, fn=get_order, fn_kwargs=kwargs, eps=1e-6, max_iter=30000)
    wl.gen_states(k=k_c, do_brute=(not do_wl))

    np.savez_compressed('hotspot_dos_p_{:02d}_q_{:02d}_kc_{:03d}'.format(p,q,k_c), sampled_points=wl.sampled_pt_idx,
                         bins=wl.bins, entropies=wl.entropies, density=wl.density, p=p, q=q, kc=k_c)


