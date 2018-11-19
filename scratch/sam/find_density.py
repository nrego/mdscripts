from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

from matplotlib import pyplot as plt

import matplotlib as mpl

from IPython import embed

import cPickle as pickle

import os

mpl.rcParams.update({'axes.labelsize': 60})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':20})

# Do exhaustive brute-force search thru states
do_brute = True
N = 36
pos_idx = np.arange(N)


k = 35
# Set up bins for density of states histogram


# Pos is a collection of points, shape (n_pts, ndim)
def get_rms(pos, prec=2):
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_diff = np.linalg.norm(diff, axis=1)**2
    return np.round( np.sqrt(sq_diff.mean()), prec)

def is_flat(hist, s=0.8):
    avg = hist.mean()

    test = hist > avg*s

    return test.all()

# Generate a new random point R_j, given existing point R_i
def trial_move(pt_idx, k):
    move_idx = np.round(np.random.normal(size=k)).astype(int)
    pt_idx_new = pt_idx + move_idx

    # Reflect new point
    pt_idx_new[pt_idx_new < 0] += 36
    pt_idx_new[pt_idx_new > 35] -= 36

    return pt_idx_new

def trial_move2(pt_idx):
    change_pts = np.random.random_integers(0,1,pt_idx.size).astype(bool)
    same_indices = pt_idx[~change_pts]
    avail_indices = np.setdiff1d(pos_idx, same_indices)
    new_pt_idx = pt_idx.copy()

    new_pt_idx[change_pts] = np.random.choice(avail_indices, change_pts.sum(), replace=False)

    return new_pt_idx

## Generate grid of center points
z_space = 0.5 # 0.5 nm spacing
y_space = np.sqrt(3)/2.0 * z_space

pos_row = np.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3.])
y_pos = 0
positions = []
for i in range(6):
    if i % 2 == 0:
        this_pos_row = pos_row
    else:
        this_pos_row = pos_row + z_space/2.0

    for j in range(6):
        z_pos = this_pos_row[j]
        positions.append(np.array([y_pos, z_pos]))

    y_pos += y_space

positions = np.array(positions)
N = positions.shape[0]


print('k: {}'.format(k))
#min_val = np.floor(10*(0.4 + (k-3)*0.05))/10.0
max_val = 1.75
rms_bins = np.arange(0, max_val+0.1, 0.05)

# Only check center bins for convergence test (flat histogram)
center_bin_lo = 0
center_bin_hi = rms_bins.size-1
center_bin_slice = slice(center_bin_lo, center_bin_hi)

# Density of states, from brute-force
states = np.zeros(rms_bins.size-1)

entropies = np.zeros_like(states)

# Indices of sampled pts...
sampled_pt_idx = np.empty(rms_bins.size-1, dtype=object)

max_rms = np.float('-inf')
min_rms = np.float('inf')
if do_brute:
    combos = np.array(list(combinations(pos_idx, k)))
    shuffle = np.random.choice(combos.shape[0], combos.shape[0], replace=False)

    for pt_idx in combos[shuffle]:
        this_pos = positions[pt_idx]

        rms = get_rms(this_pos)
        if rms > max_rms:
            max_rms = rms
        elif rms < min_rms:
            min_rms = rms

        #print("combo: {}".format(this_pos_idx))
        #print("  rms: {}".format(rms))

        bin_assign = np.digitize(rms, rms_bins) - 1
        states[bin_assign] += 1

        this_arr = sampled_pt_idx[bin_assign]
        if this_arr is None:
            sampled_pt_idx[bin_assign] = np.array([pt_idx])

        elif this_arr.shape[0] < 10:
            this_arr = np.vstack((this_arr, pt_idx))
            sampled_pt_idx[bin_assign] = np.unique(this_arr, axis=0)

    print('  Max rms: {}'.format(max_rms))
    print('  Min rms: {}'.format(min_rms))

    states /= np.diff(rms_bins)
    states /= np.dot(np.diff(rms_bins), states)

# Do Wang-Landau

else:
    
    max_iter = 60000
    n_iter = 0
    eps = 10**(-4)
    wl_entropies = np.zeros_like(states)
    wl_hist = np.zeros_like(states)

    pt_idx = np.sort( np.random.choice(pos_idx, size=k, replace=False) )
    pt = positions[pt_idx]

    rms = get_rms(pt)
    bin_assign = np.digitize(rms, rms_bins) - 1

    f = 1
    wl_entropies[bin_assign] += f
    wl_hist[bin_assign] += 1

    M_iter = 0
    print("M: {}".format(M_iter))
    while f > eps:
        n_iter += 1


        pt_idx_new = np.sort( trial_move2(pt_idx) )
        pt_new = positions[pt_idx_new]

        rms_new = get_rms(pt_new)

        assert np.unique(pt_idx_new).size == k

        bin_assign_new = np.digitize(rms_new, rms_bins) - 1

        # Accept trial move
        if np.log(np.random.random()) < (wl_entropies[bin_assign] - wl_entropies[bin_assign_new]):
            pt_idx = pt_idx_new
            pt = pt_new
            rms = rms_new
            bin_assign = bin_assign_new

        # Update histogram and density of states
        wl_entropies[bin_assign] += f
        wl_hist[bin_assign] += 1
        states[bin_assign] += 1
        
        this_arr = sampled_pt_idx[bin_assign]
        if this_arr is None:
            sampled_pt_idx[bin_assign] = np.array([pt_idx])

        elif this_arr.shape[0] < 10:
            this_arr = np.vstack((this_arr, pt_idx))
            sampled_pt_idx[bin_assign] = np.unique(this_arr, axis=0)

        # update the center_bin_slice by moving up lower bin boundary, if necessary
        if n_iter > 0.5*max_iter:
            
            occupied_idx = np.argwhere(wl_hist > 0)
            center_bin_lo = occupied_idx.min()
            center_bin_hi = occupied_idx.max()+1
            if center_bin_hi -center_bin_lo == 1:
                break
            center_bin_slice = slice(center_bin_lo, center_bin_hi)

        if is_flat(wl_hist[center_bin_slice], 0.7) or n_iter > max_iter:

            #break
            print(" n_iter: {}".format(n_iter))
            n_iter = 0
            prev_hist = wl_hist.copy()
            wl_hist[:] = 0
            f = 0.5 * f
            M_iter += 1
            print("M : {}".format(M_iter))

    wl_entropies -= wl_entropies.max()
    wl_states = np.exp(wl_entropies)
    wl_states /= np.diff(rms_bins)
    wl_states /= np.dot(np.diff(rms_bins), wl_states)

occupied_idx = states > 0

## Print out indices for each point at each rms
dirname = 'k_{:02d}'.format(k)
os.makedirs(dirname)

fout = open('{}/pt_idx_data.pkl'.format(dirname), 'w')
output_payload = (rms_bins, occupied_idx, positions, sampled_pt_idx)
pickle.dump(output_payload, fout)


