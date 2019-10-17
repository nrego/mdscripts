from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

from matplotlib import pyplot as plt

import matplotlib as mpl

from IPython import embed

from scratch.sam.util import *
import os

## Generate patterns for small patches, either by hand or with WL (according to predicted energy using model 3 for the 6x6 patch)

parser = argparse.ArgumentParser('Generate patterns (indices) for a square patch size')
parser.add_argument('--do-wl', action='store_true',
                    help='If true, generate patterns using WL algorithm')
parser.add_argument('--patch-size', default=2, type=int,
                    help='Size of patch side (total number of head groups is patch_size**2) (default: %(default)s)')
parser.add_argument('--k-ch3', default=0, type=int,
                    help='Number of methyls for this patch type (default: %(default)s)')
parser.add_argument('--sam-data', default='')
args = parser.parse_args()


# Do exhaustive brute-force search thru states
do_brute = not args.do_wl
patch_size = args.patch_size
N = patch_size**2
k = args.k_ch3
assert k <= N

# Internal indexing system for each patch positin
pos_idx = np.arange(N)


## Generate grid of patch points
z_space = 0.5 # 0.5 nm spacing
y_space = np.sqrt(3)/2.0 * z_space

pos_row = np.array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3.])
y_pos = 0
positions = []
for i in range(patch_size):
    if i % 2 == 0:
        this_pos_row = pos_row
    else:
        this_pos_row = pos_row + z_space/2.0

    for j in range(patch_size):
        z_pos = this_pos_row[j]
        positions.append(np.array([y_pos, z_pos]))

    y_pos += y_space

positions = np.array(positions)
assert N == positions.shape[0]


print('k: {}'.format(k))
#min_val = np.floor(10*(0.4 + (k-3)*0.05))/10.0

if do_brute:
    combos = np.array(list(combinations(pos_idx, k)))
    #shuffle = np.random.choice(combos.shape[0], combos.shape[0], replace=False)

    for pt_idx in combos:
        # Positions of all methyls
        this_pos = positions[pt_idx]

        bin_assign = np.digitize(rms, rms_bins) - 1
        states[bin_assign] += 1

        this_arr = sampled_pt_idx[bin_assign]
        if this_arr is None:
            sampled_pt_idx[bin_assign] = np.array([pt_idx])

        elif this_arr.shape[0] < 10:
            this_arr = np.vstack((this_arr, pt_idx))
            sampled_pt_idx[bin_assign] = np.unique(this_arr, axis=0)



## Print out indices for each point at each rms
dirname = 'k_{:02d}'.format(k)

try:
    os.makedirs(dirname)
except OSError:
    print("directory {} already exists - exiting".format(dirname))
    exit()

with open('{}/pt_idx_data.pkl'.format(dirname), 'w') as fout:
    output_payload = (rms_bins, occupied_idx, positions, sampled_pt_idx)
    pickle.dump(output_payload, fout)

