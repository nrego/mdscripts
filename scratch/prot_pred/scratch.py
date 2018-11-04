from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import MDAnalysis

import os, glob
from mdtools import dr
from IPython import embed

from system import MDSystem

from rhoutils import rho, cartesian
from fieldwriter import RhoField

from scipy.spatial import cKDTree

import argparse


univ = MDAnalysis.Universe('equil.tpr', 'cent.xtc')

box = np.array([60., 60., 120.])
# slab width in the z dimension
slab_width = 1.0
n_slabs = box[-1] / slab_width
slabs_z = np.arange(0, box[-1]+slab_width, slab_width)

first_frame = 20000
last_frame = univ.trajectory.n_frames
n_frames = last_frame - first_frame

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


int_pos = gibbs_int_pos.mean()
min_dist = np.zeros((n_frames, prot_h.n_atoms))
for idx, i_frame in enumerate(range(first_frame, last_frame)):
    univ.trajectory[idx]

    this_z_pos = prot_h.positions[:,2]
    dist = int_pos - this_z_pos
    dist[dist<0] = 0
    min_dist[idx] = dist
'''
# do instantaneous interface 
grid_min_z = 50
grid_res = 1.0

pts_x = np.arange(0, box[0], grid_res)
pts_y = np.arange(0, box[1], grid_res)
pts_z = np.arange(grid_min_z, box[2], grid_res)

gridpts = cartesian([pts_x, pts_y, pts_z]).astype(np.float32)
npts = gridpts.shape[0]
tree_grid = cKDTree(gridpts)
rho_vals = np.zeros(npts)

r_cut = 7.0
r_cut_sq = r_cut**2
sigma = 2.4
sigma_sq = sigma**2

water_pos = waters.positions
water_mask = water_pos[:,2] > grid_min_z

tree_water = cKDTree(water_pos[water_mask])
res = tree_water.query_ball_tree(tree_grid, r_cut, p=np.inf)
# For each water ...
for i, pos in enumerate(water_pos[water_mask]):
    neighbor_idx = np.array(res[i])

    dist_vector = gridpts[neighbor_idx] - pos

    this_rho = rho(dist_vector, sigma, sigma_sq, r_cut, r_cut_sq)

    rho_vals[neighbor_idx] += this_rho

rho_vals /= 0.033
rho_shape = rho_vals.reshape((1, pts_x.size, pts_y.size, pts_z.size))

field = RhoField(rho_shape, gridpts)
'''