from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import MDAnalysis

import os, glob
from mdtools import dr
from IPython import embed

from system import MDSystem

import argparse


univ = MDAnalysis.Universe('equil.tpr', 'cent.xtc')

box = np.array([60., 60., 120.])
# slab width in the z dimension
slab_width = 1.0
n_slabs = box[-1] / slab_width
slabs_z = np.arange(0, box[-1]+slab_width, slab_width)

first_frame = 20000
last_frame = univ.trajectory.n_frames

prot = univ.select_atoms('segid seg_0_Protein_targ')
prot_h = prot.select_atoms('not name H*')

## Find bounding square in x-y plane around hfb 
  #(so we don't consider waters in this region when calculating interface)
lim_min = np.array([np.inf, np.inf])
lim_max = np.array([-np.inf, -np.inf])

for i_frame in range(first_frame, last_frame):
    univ.trajectory[i_frame]

    min_x, min_y, min_z = prot.positions.min(axis=0)
    max_x, max_y, max_z = prot.positions.max(axis=0)

    if min_x < lim_min[0]:
        lim_min[0] = min_x
    if min_y < lim_min[1]:
        lim_min[1] = min_y

    if max_x > lim_max[0]:
        lim_max[0] = max_x
    if max_y > lim_max[1]:
        lim_max[1] = max_y


lim_min = np.floor(lim_min)
lim_max = np.ceil(lim_max)

waters = univ.select_atoms('name OW')

# z coord of the gibs dividing interface
z_int = np.zeros((last_frame-first_frame, slabs_z.size-1))

small_y_ht = lim_max[1] - lim_min[1]
slab_vol = ((box[0]*box[1]) - (lim_max - lim_min).prod()) * slab_width
expt_water = slab_vol * 0.033
for idx, i_frame in enumerate(range(first_frame, last_frame)):
    univ.trajectory[i_frame]

    water_pos = waters.positions
    water_mask = ((water_pos[:,0] < lim_min[0]) | (water_pos[:,0] > lim_max[0])) | ((water_pos[:,1] < lim_min[1]) | (water_pos[:,1] > lim_max[1]))

    water_pos = water_pos[water_mask]
    counts, bb = np.histogram(water_pos[:,2], bins=slabs_z)

    z_int[idx] = counts/expt_water


gibbs_int_pos = np.zeros(z_int.shape[0])
# Find the gibs dividing surface (upper) for each frame
for idx in range(z_int.shape[0]):
    this_density = z_int[idx]
    
    mask = this_density < 0.5
    density_mask = mask[:-1] != mask[1:]
    int_idx = np.where(density_mask==True)[0]

    if int_idx.size > 2:
        print("idx: {}, int_idx: {}".format(idx, int_idx))

    gibbs_int_pos[idx] = slabs_z[int_idx[-1]]


