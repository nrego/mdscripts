
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


def get_energy(pt_idx, m_mask, nn, reg):
    
    coef1, coef2, coef3 = reg.coef_
    #inter = reg.intercept_

    k_m = m_mask.sum()
    n_mm = 0
    n_mo = 0

    for m_idx in pt_idx:
        for n_idx in nn[m_idx]:
            if n_idx > m_idx:
                n_mm += m_mask[n_idx]
            n_mo += ~m_mask[n_idx]
    
    return coef1*k_m + coef2*n_mm + coef3*n_mo

# Pos is a collection of points, shape (n_pts, ndim)
def get_rms(pt_idx, m_mask, positions, prec=2):
    pos = positions[m_mask]
    if pos.size == 0:
        return 0.0
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_dev = (diff**2).sum(axis=1)

    return np.round(np.sqrt(sq_dev.mean()), prec)


## Generate patterns for small patches, either by hand or with WL (according to predicted energy using model 3 for the 6x6 patch)

parser = argparse.ArgumentParser('Generate patterns (indices) for a square patch size')
parser.add_argument('--do-wl', action='store_true',
                    help='If true, generate patterns using WL algorithm')
parser.add_argument('--do-rms', action='store_true',
                    help='If true, use pattern RMS as order parameter for WL/state generation (otherwise, use M3 energies)')
parser.add_argument('--patch-size', default=1, type=int,
                    help='Size of patch side (total number of head groups is patch_size**2) (default: %(default)s)')
parser.add_argument('--patch-size-2', default=None, type=int,
                    help='Size of other patch dimension (default is same as patch_size)')
parser.add_argument('--k-ch3', default=0, type=int,
                    help='Number of methyls for this patch type (default: %(default)s)')
parser.add_argument('--sam-data', default='../sam_pattern_data.dat.npz')
parser.add_argument('--eps', type=float, default='1e-10',
                    help='Epsilon (tolerance) for Wang Landau (default: 1e-10)')
parser.add_argument('--max-wl-iter', type=int, default=60000,
                    help='Maximum number of WL MC steps to take each iteration (default: %(default)s)')
parser.add_argument('--hist-flat-tol', type=float, default=0.8,
                    help='Criterion for determining if histogram is flat for each \
                          Wang-Landau iteration (default: %(default)s)')
args = parser.parse_args()

########## Generate M3 ##########
#################################
energies, methyl_pos, k_vals, positions, pos_ext, patch_indices, nn, nn_ext, edges, ext_indices, int_indices = extract_data(args.sam_data)

#n_mm, n_oo, n_mo, n_me, n_oe
k_eff = np.zeros((energies.size, 5))
for i, methyl_mask in enumerate(methyl_pos):
    k_eff[i] = get_keff_all(methyl_mask, edges, patch_indices).sum(axis=0)
n_mm, n_oo, n_mo, n_me, n_oe = np.split(k_eff, 5, axis=1)


feat = np.hstack((k_vals[:,None], n_mm, n_mo))
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat, energies, sample_weight=np.ones_like(energies))

######### DONE GEN M3 ###########
#################################


# Do exhaustive brute-force search thru states
do_brute = not args.do_wl
patch_size = args.patch_size
if args.patch_size_2 is not None:
    patch_size_2 = args.patch_size_2
else:
    patch_size_2 = patch_size

N = patch_size*patch_size_2
k_ch3 = args.k_ch3
assert k_ch3 <= N


# Internal indexing system for each patch positin
pos_idx = np.arange(N)

state = State(np.arange(N), patch_size, patch_size_2)
positions = state.positions
assert N == positions.shape[0]
pos_ext = state.pos_ext

# nn will be used for estimating energies
nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

# Find minimum energy (i.e. when everything's methyl)
if not args.do_rms:
    min_e = get_energy(pos_idx, np.ones(N).astype(bool), nn, reg)
    bins = np.arange(np.floor(min_e), 2, 1)

    order_fn = get_energy
    fn_kwargs = dict(nn=nn, reg=reg)

else:

    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    test_pos = np.vstack((min_pos, max_pos)) 

    # Hackish, but gets maximum rms we can expect
    max_rms = get_rms(None, np.ones(2, dtype=bool), test_pos)

    bins = np.arange(0.0, max_rms+0.1, 0.05)

    order_fn = get_rms
    fn_kwargs = dict(positions=positions)


print('k: {}'.format(k_ch3))
print('{} bins from {} to 0'.format(bins.size-1, bins.min()))


######### RUN WL ##########
###########################


wl = WangLandau(positions, bins, order_fn, fn_kwargs=fn_kwargs, eps=args.eps, max_iter=args.max_wl_iter)

wl.gen_states(k_ch3, do_brute, hist_flat_tol=args.hist_flat_tol)

np.savez_compressed('rmsd_dos_p_{:02d}_q_{:02d}_kc_{:03d}'.format(6,6,k_c), sampled_points=wl.sampled_pt_idx,
                     bins=wl.bins, entropies=wl.entropies, density=wl.density,  kc=k_c)

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


