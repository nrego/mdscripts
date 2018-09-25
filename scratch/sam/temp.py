from __future__ import division, print_function

import numpy as np
from scipy.special import binom
from itertools import combinations

import matplotlib as mpl

from matplotlib import pyplot as plt


mpl.rcParams.update({'axes.labelsize': 60})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':40})


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

#indices = [8,9,10,13,14,15, 20, 21, 22]
indices = [8,10,14,15, 20, 21, 22,25, 33]
indices = np.random.choice(pos_idx, 9, replace=True)

pos_cent = positions[indices]

print("RMS: {}".format(get_rms(pos_cent)))

fig, ax = plt.subplots(figsize=(5,4.8))

ax.plot(positions[:,0], positions[:,1], 'bo', markersize=20)
ax.plot(pos_cent[:,0], pos_cent[:,1], 'ko', markersize=20)

ax.set_xlim(-0.5, 2.7)
ax.set_xticks([0,1.0,2.0])
ax.set_ylim(-0.5, 2.7)
ax.set_yticks([0,1.0,2.0])

#ax.set_xlabel(r'$y$')
#ax.set_ylabel(r'$z$')
plt.tight_layout()
plt.show()
