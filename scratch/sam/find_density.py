from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

# Do exhaustive brute-force search thru states
do_brute = True

# Pos is a collection of points, shape (n_pts, ndim)
def get_rms(pos):
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_diff = np.linalg.norm(diff, axis=1)**2
    return np.sqrt(sq_diff.mean())

def is_flat(hist, s=0.8):
    avg = hist.mean()

    test = hist > avg*s

    return test.all()

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
        this_pos_row = pos_row - z_space/2.0

    for j in range(6):
        z_pos = this_pos_row[j]
        positions.append(np.array([y_pos, z_pos]))

    y_pos += y_space

positions = np.array(positions)
N = positions.shape[0]

pos_idx = np.arange(N)

# Set up bins for density of states histogram
k = 5
min_val = np.floor(10*(0.4 + (k-3)*0.05))/10.0
max_val = 1.7
rms_bins = np.arange(min_val, max_val, 0.05)
rms_bins = np.append(0, rms_bins)
rms_bins = np.append(rms_bins, positions.max())

# Density of states, from brute-force
states = np.zeros(rms_bins.size-1)

entropies = np.zeros_like(states)



max_rms = np.float('-inf')
min_rms = np.float('inf')
if do_brute:
    combos = np.array(list(combinations(pos_idx, k)))

    for this_pos_idx in combos:
        this_pos = positions[list(this_pos_idx)]

        rms = get_rms(this_pos)
        if rms > max_rms:
            max_rms = rms
        elif rms < min_rms:
            min_rms = rms

        #print("combo: {}".format(this_pos_idx))
        #print("  rms: {}".format(rms))

        bin_assign = np.digitize(rms, rms_bins) - 1
        states[bin_assign] += 1

    print('Max rms: {}'.format(max_rms))
    print('Min rms: {}'.format(min_rms))

    states /= np.diff(rms_bins)
    states /= np.dot(np.diff(rms_bins), states)
# Do Wang-Landau

else:
    max_iter = 10000000
    n_iter = 0
    eps = 10**(-8)
    wl_entropies = np.zeros_like(states)
    wl_hist = np.zeros_like(states)

    pt_idx = np.random.choice(pos_idx, size=k, replace=False)
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

        move_idx = np.round(np.random.normal(size=k)).astype(int)
        pt_idx_new = pt_idx + move_idx

        # Reflect new point
        pt_idx_new[pt_idx_new < 0] += 36
        pt_idx_new[pt_idx_new > 35] -= 36

        pt_new = positions[pt_idx_new]
        rms_new = get_rms(pt_new)

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

        if n_iter > max_iter:
            break
        if is_flat(wl_hist[1:-1], 0.85) or n_iter > max_iter:
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
