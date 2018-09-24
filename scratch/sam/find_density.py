from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations


# Pos is a collection of points, shape (n_pts, ndim)
def get_rms(pos):
    centroid = pos.mean(axis=0)
    diff = pos-centroid
    sq_diff = np.linalg.norm(diff, axis=1)**2
    return np.sqrt(sq_diff.mean())

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
rms_bins = np.arange(0, positions.max(), 0.01)
states = np.zeros(rms_bins.size-1)

k = 5

combos = np.array(list(combinations(pos_idx, k)))

for this_pos_idx in combos:
    this_pos = positions[list(this_pos_idx)]

    rms = get_rms(this_pos)
    #print("combo: {}".format(this_pos_idx))
    #print("  rms: {}".format(rms))

    bin_assign = np.digitize(rms, rms_bins) - 1
    states[bin_assign] += 1

