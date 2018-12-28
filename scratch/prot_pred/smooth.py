from __future__ import division, print_function

import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

from constants import k

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
args = parser.parse_args()

smooth_width = args.smooth_width

rho_ref_data = np.load(args.rho_ref)['rho_water'].mean(axis=0)
assert rho_ref_data.ndim == 1

rhos = np.array(sorted(glob.glob(args.rho_files)))

phis = np.array([float(rho.split('/')[3].split('_')[1]) for rho in rhos]) / 10.0
beta_phis = beta * phis

assert beta_phis[0] == 0
rhos[0] = args.rho_ref # Make sure our beta*phi=0 goes back to rho_data_dump from equil directory (i.e. the reference)

all_indices = np.arange(len(beta_phis))
for i, beta_phi in enumerate(beta_phis):
    diffs = np.abs(beta_phi - beta_phis)
    #indices_diffs = np.abs(i - all_indices)

    # Indices of all beta*phi vals that are within window
    smooth_window_indices = diffs <= smooth_width

    smooth_fnames = rhos[smooth_window_indices]

    smooth_rho = np.empty((smooth_fnames.size, rho_ref_data.size))

    for j, smooth_fname in enumerate(smooth_fnames):
        this_rho = np.load(smooth_fname)['rho_water'].mean(axis=0)
        smooth_rho[j, ...] = this_rho


    #smooth_rho = smooth_rho.mean(axis=0)

    dirname = 'phi_{:03d}'.format(int(10*phis[i]))
    try:
        os.makedirs(dirname)
    except OSError:
        pass

    np.savez_compressed('{}/rho_data_dump_smooth.dat'.format(dirname), rho_water=smooth_rho, phi_vals=smooth_fnames, 
                        header='n_i from smoothed data sets')


