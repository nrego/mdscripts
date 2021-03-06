from __future__ import division, print_function

import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

from constants import k

import scipy

import MDAnalysis

import argparse
from IPython import embed

beta = 1/(300*k)

parser = argparse.ArgumentParser('produce smoothed rho values for each phi (excluding beta*phi=0)')
parser.add_argument('--rho-ref', type=str, default='../equil/rho_data_dump_rad_6.0.dat.npz',
                    help='Location of reference <n_i>_0 values (default: %(default)s')
parser.add_argument('--rho-files', type=str, default='../prod/phi_sims/phi_*/rho_data_dump_rad_6.0.dat.npz',
                    help='Glob-like string to determine <n_i>_\phi values (default: %(default)s')
parser.add_argument('--smooth-width', type=float, default=0.1,
                    help='Smoothing width - will use beta*phi results that are at most this far from current beta*phi when calculating <\sigma_i>_phi')
parser.add_argument('--thresh', type=float, default=0.5,
                    help='Threshold for indicator function (determining if dewet or not)')
args = parser.parse_args()

smooth_width = args.smooth_width
thresh = args.thresh

n_0_data = np.load(args.rho_ref)['rho_water'].mean(axis=0)
assert n_0_data.ndim == 1

n_phi_fnames = np.array(sorted(glob.glob(args.rho_files)))

phis = np.array([float(fname.split('/')[3].split('_')[1]) for fname in n_phi_fnames]) / 10.0
beta_phis = beta * phis
n_phis = beta_phis.size

assert beta_phis[0] == 0
n_phi_fnames[0] = args.rho_ref # Make sure our beta*phi=0 goes back to rho_data_dump from equil directory (i.e. the reference)

all_indices = np.arange(len(beta_phis))
beta_phi_max = beta_phis.max()
for i, beta_phi in enumerate(beta_phis):

    # Find other phi values that are within the smoothing window
    diffs = np.abs(beta_phi - beta_phis)
    smooth_window_indices = diffs <= smooth_width

    # Sanity - smoothing better include current value of phi!
    assert smooth_window_indices[i]

    windows_right = smooth_window_indices[i+1:].sum()
    windows_left = smooth_window_indices[::-1][n_phis-i:].sum()
    print("i: {} windows_left: {} windows_right: {}".format(i, windows_left, windows_right))
    smooth_fnames = n_phi_fnames[smooth_window_indices]
    smooth_beta_phis = beta_phis[smooth_window_indices]
    #embed()
    if windows_left:
        smooth_beta_phis = np.append(beta_phi-smooth_width, smooth_beta_phis)
    if windows_right:
        smooth_beta_phis = np.append(smooth_beta_phis, beta_phi+smooth_width)

    if windows_left + windows_right == 0:
        weights = np.array([1.])
    else:
        weights_right = np.diff(smooth_beta_phis[windows_left:])
        weights_left = np.diff(smooth_beta_phis[:windows_left])
        weights = np.append(weights_left, weights_right)
        weights /= weights.sum()
        weights *= smooth_window_indices.sum()

    # Technically this will be the smoothed n_i_\phi's
    smooth_rho = np.empty((smooth_fnames.size, n_0_data.size))
    
    smooth_h = np.empty_like(smooth_rho)

    for j, smooth_fname in enumerate(smooth_fnames):
        this_n_phi = np.load(smooth_fname)['rho_water'].mean(axis=0) 
        this_rho = this_n_phi / n_0_data
        smooth_rho[j, ...] = this_n_phi

        this_h = (this_rho >= thresh).astype(float)
        smooth_h[j, ...] = this_h * n_0_data

    dirname = 'phi_{:03d}'.format(int(10*phis[i]))
    try:
        os.makedirs(dirname)
    except OSError:
        pass

    try:
        smooth_rho = smooth_rho * weights[:, None]
    except:
        embed()

    smooth_h = smooth_h * weights[:, None]
    np.savez_compressed('{}/rho_data_dump_smooth.dat'.format(dirname), rho_water=smooth_rho, phi_vals=smooth_fnames, weights=weights,
                        header='n_i from smoothed data sets')
    np.savez_compressed('{}/h_data_dump_smooth.dat'.format(dirname), rho_water=smooth_h, phi_vals=smooth_fnames, weights=weights,
                        header='h_i*n_0 from smoothed data sets')

