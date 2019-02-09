from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Template for analyzing mu_ex for all patch patterns
homedir = os.environ['HOME']
ds_k = np.load('pattern_sample/analysis_data.dat.npz')
ds_l = np.load('inv_pattern_sample/analysis_data.dat.npz')

positions = ds_k['positions']
assert np.array_equal(positions, ds_l['positions'])

rms_bins = ds_k['rms_bins']
assert np.array_equal(rms_bins, ds_l['rms_bins'])

k_bins = ds_k['k_bins']
assert np.array_equal(k_bins, ds_l['k_bins'])

xx, yy = np.meshgrid(k_bins[:-1], rms_bins[:-1])
k_vals_unpack = l_vals = xx.ravel()
rms_vals_unpack = yy.ravel()

# Now unpack and combine all data...
dat_k = ds_k['energies'].ravel()
mask_k = ~np.ma.masked_invalid(dat_k).mask
dat_k = dat_k[mask_k]
k_vals_k = k_vals_unpack[mask_k]
rms_vals_k = rms_vals_unpack[mask_k]
methyl_pos_k = ds_k['methyl_mask'].reshape(xx.size, 36)[mask_k,:]

dat_l = ds_l['energies'].ravel()
mask_l = ~np.ma.masked_invalid(dat_l).mask
dat_l = dat_l[mask_l]
k_vals_l = 36 - k_vals_unpack[mask_l]
# Need to get the D_ch3 values from positions...

dat_pooled = np.append(dat_k, dat_l)
k_vals = np.append(k_vals_k, k_vals_l)


