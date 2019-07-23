from __future__ import division, print_function

import numpy as np

import argparse
import logging

import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed

from scratch.voxel_2pt.gridutils import *

## Use final mask to output voxels ##
dat = np.load("phi_sims/ni_weighted.dat.npz")

beta_phis = dat['beta_phi']
n_with_phi = dat['avg']
n_0 = n_with_phi[:,0]

red_mask = np.loadtxt("red_mask.dat", dtype=bool)

ds = np.load('phi_sims/phi_000/init_data.pkl.npz')
gridpt_mask = ds['gridpt_mask']
x_bounds = ds['x_bounds']
y_bounds = ds['y_bounds']
z_bounds = ds['z_bounds']

gridpts = construct_gridpts(x_bounds, y_bounds, z_bounds)

n_pts_red = gridpt_mask.sum()

gridpts_red = gridpts[gridpt_mask]

dg_water = np.loadtxt('2pt/dg_voxel.dat')
# kj/mol per A^3
dg_vol = np.loadtxt('2pt/dg_vol_voxel.dat') / 1000.
beta_phi_star = np.loadtxt('beta_phi_star.dat')

assert dg_water.size == dg_vol.size == beta_phi_star.size == n_pts_red

save_gridpts('dg_vol.pdb', gridpts_red[red_mask], dg_vol[red_mask])
save_gridpts('dg_water.pdb', gridpts_red[red_mask], dg_water[red_mask])
save_gridpts('beta_phi_star.pdb', gridpts_red[red_mask], beta_phi_star[red_mask])



