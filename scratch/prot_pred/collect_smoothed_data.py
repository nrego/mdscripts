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
##   run this script from the same dir (smooth_[]_dir/) to collect all the data dumps (i.e. the smoothed <n_i>_\phi's)
#    and place them in one large array to give it as function of phi
beta = 1/(300*k)

buried_mask = np.loadtxt('../bound/buried_mask.dat', dtype=bool)
surf_mask = ~buried_mask

actual_contacts = np.loadtxt('../bound/actual_contact_mask.dat', dtype=bool)
assert (actual_contacts[buried_mask] == 0).all()
actual_contacts = actual_contacts[surf_mask]

fnames = sorted( glob.glob('phi_*/rho_data_dump_smooth.dat.npz') )
ref = np.load('../equil/rho_data_dump_rad_6.0.dat.npz')['rho_water'].mean(axis=0)
assert (ref[buried_mask] < 5).all()
assert (ref[surf_mask] >= 5).all()

ref = ref[surf_mask]

phi_vals = np.array([ float(fname.split('/')[0].split('_')[-1]) for fname in fnames ]) / 10.0
beta_phi_vals = beta*phi_vals
# Sanity - in sorted order
assert (np.diff(phi_vals) > 0).all()

rho_i_with_phi = np.zeros((beta_phi_vals.size, surf_mask.sum()))
h_i_with_phi = np.zeros_like(rho_i_with_phi)

for idx, fname in enumerate(fnames):
    subdir = os.path.dirname(fname)
    n_phi = np.load(fname)['rho_water'].mean(axis=0)[surf_mask]
    rho_phi = n_phi / ref
    rho_i_with_phi[idx, :] = rho_phi

    thresh_fname = '{}/h_data_dump_smooth.dat.npz'.format(subdir)
    h_phi = np.load(thresh_fname)['rho_water'].mean(axis=0)[surf_mask] / ref
    h_i_with_phi[idx, :] = h_phi

np.savez_compressed('rho_i_with_phi.dat', rho_i=rho_i_with_phi, beta_phi=beta_phi_vals)
np.savez_compressed('h_i_with_phi.dat', h_i=h_i_with_phi, beta_phi=beta_phi_vals)
