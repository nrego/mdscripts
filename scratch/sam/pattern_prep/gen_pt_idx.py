
import numpy as np
from scipy.special import binom
from itertools import combinations

from matplotlib import pyplot as plt

import matplotlib as mpl

from IPython import embed

from scratch.sam.util import *
from wang_landau import WangLandau
import os

import pickle



# Pos is a collection of points, shape (n_pts, ndim)
def get_rms(pt_idx, m_mask, positions, prec=2):
    pos = positions[m_mask]
    if pos.size == 0:
        return [0.0]
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_dev = (diff**2).sum(axis=1)

    return [np.round(np.sqrt(sq_dev.mean()), prec)]


## Generate patterns for small patches, either by hand or with WL (according to predicted energy using model 3 for the 6x6 patch)

parser = argparse.ArgumentParser('Generate patterns (indices) for an arbitrary patch size')
parser.add_argument('--do-wl', action='store_true',
                    help='If true, generate patterns using WL algorithm')
parser.add_argument('-p', '--patch-size', default=1, type=int,
                    help='Size of patch side, p (default: %(default)s)')
parser.add_argument('-q', '--patch-size-2', default=None, type=int,
                    help='Size of other patch dimension, q (default is same as patch_size, p)')
parser.add_argument('--k-ch3', default=0, type=int,
                    help='Number of methyls for this patch type (default: %(default)s)')
parser.add_argument('--eps', type=float, default='1e-10',
                    help='Epsilon (tolerance) for Wang Landau (default: 1e-10)')
parser.add_argument('--max-wl-iter', type=int, default=60000,
                    help='Maximum number of WL MC steps to take each iteration (default: %(default)s)')
parser.add_argument('--hist-flat-tol', type=float, default=0.8,
                    help='Criterion for determining if histogram is flat for each \
                          Wang-Landau iteration (default: %(default)s)')
args = parser.parse_args()



# Do exhaustive brute-force search thru states
do_brute = not args.do_wl
p = patch_size = args.patch_size
if args.patch_size_2 is not None:
    q = patch_size_2 = args.patch_size_2
else:
    q = patch_size_2 = patch_size

N = p*q
k_ch3 = args.k_ch3
assert k_ch3 <= N


# Internal indexing system for each patch position
pos_idx = np.arange(N)

# Make a dummy state to extract positions, pos_ext
state = State(np.arange(N), p, q)
positions = state.positions
assert N == positions.shape[0]
pos_ext = state.pos_ext

min_pos = positions.min(axis=0)
max_pos = positions.max(axis=0)
test_pos = np.vstack((min_pos, max_pos)) 

# Hackish, but gets maximum rms we can expect
max_rms = get_rms(None, np.ones(2, dtype=bool), test_pos)[0]

bins = np.arange(0.0, max_rms+0.1, 0.05)

order_fn = get_rms
fn_kwargs = dict(positions=positions)


print('k: {}'.format(k_ch3))
print('{} bins from {} to 0'.format(bins.size-1, bins.min()))


######### RUN WL ##########
###########################


wl = WangLandau(positions, [bins], order_fn, fn_kwargs=fn_kwargs, eps=args.eps, max_iter=args.max_wl_iter)

wl.gen_states(k_ch3, do_brute, hist_flat_tol=args.hist_flat_tol)


## Print out indices for each point at each rms
dirname = 'k_{:02d}'.format(k_ch3)

try:
    os.makedirs(dirname)
except OSError:
    print("directory {} already exists - exiting".format(dirname))
    #exit()

print("saving payload for k={} with patch size {} by {}".format(k_ch3, patch_size, patch_size))
with open('{}/pt_idx_data.pkl'.format(dirname), 'wb') as fout:
    output_payload = (wl.bins, wl.density>0, wl.positions, wl.sampled_pt_idx)
    pickle.dump(output_payload, fout)


