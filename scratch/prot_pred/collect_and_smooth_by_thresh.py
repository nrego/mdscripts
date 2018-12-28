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

## After running smooth.py (which deposits a new rho_data_dump_smooth in each phi dir of smooth_[]_pred/)
##   run this script from 2tsc directory to collect all the <n_i>_\phis, find the thresholds (i.e. when they're < 0.5),
#    and then smooth all at once with various smoothing windows
#
#  Will make a new file in each smooth_[]_pred directory ('thresh_i_with_phi')

beta = 1/(300*k)

fnames = sorted(glob.glob('smooth_*/rho_i_with_phi.dat.npz'))
delta_beta_phis = np.array([float(fname.split('/')[0].split('_')[-2]) for fname in fnames]) / 100.0
assert delta_beta_phis[0] == 0

# (unsmoothed) dataset that contains every <rho_i>_\phi for each surface atom
dat = np.load('smooth_000_pred/rho_i_with_phi.dat.npz')
rho_i_with_phi = dat['rho_i']
beta_phis = dat['beta_phi']
# shape: (n_beta_phi_vals, n_surf_atoms)
thresh = (rho_i_with_phi >= 0.5).astype(float)

## Same algorithm as smooth.py
for idx, delta_beta_phi in enumerate(delta_beta_phis):

    smooth_thresh = np.zeros(())
    # Grab each h_i_\phi for each phi
    for i, beta_phi in enumerate(beta_phis):
        diffs = np.abs(beta_phi - beta_phis)
        smooth_window_indices = diffs <= delta_beta_phi