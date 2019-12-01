from __future__ import division, print_function

import numpy as np

import argparse
import logging


import os, glob, sys

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed

from whamutils import get_negloghist, extract_and_reweight_data


dat = np.load('phi_sims/ni_rad_weighted.dat.npz')

all_data_ds = np.load('phi_sims/data_reduced/all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_aux']

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)
bins = np.arange(0, max_val+1, 1)

## In kT!
beta_phi_vals = np.linspace(0,4,101)

## Get PvN, <Nv>, chi_v from all data ###
all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_N, all_chi_N, _ = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)


### Now input all <n_i>_\phi's for a given i ###
print('')
print('Extracting all n_i\'s...')
sys.stdout.flush()

n_heavies = None

n_i_dat_fnames = np.append(sorted(glob.glob('phi_sims/data_reduced/phi*/rho_data_dump_rad_6.0.dat.npz')), sorted(glob.glob('phi_sims/data_reduced/nstar*/rho_data_dump_rad_6.0.dat.npz')))
all_data_n_i = None

## Gather n_i data from each umbrella window (phi value)
for fname in n_i_dat_fnames:

    ## Shape: (n_heavies, n_frames) ##
    n_i = np.load(fname)['rho_water'].T

    if n_heavies is None:
        n_heavies = n_i.shape[0]
    else:
        assert n_i.shape[0] == n_heavies

    if all_data_n_i is None:
        all_data_n_i = n_i.copy()
    else:
        all_data_n_i = np.append(all_data_n_i, n_i, axis=1)

print('...Done.')
print('')


### START COV ANALYSIS ###
data = all_data_n_i
ntwid = all_data
logweights = all_logweights
beta_phi_val = 2.40

print("beta phi: {:.2f}".format(beta_phi_val))
sys.stdout.flush()
bias_logweights = logweights - beta_phi_val*ntwid
bias_logweights -= bias_logweights.max()
norm = np.log(np.sum(np.exp(bias_logweights)))
bias_logweights -= norm

bias_weights = np.exp(bias_logweights)

avg_data = np.dot(data, bias_weights)
centered_data = data - avg_data[:, None]

this_cov = np.multiply(centered_data, bias_weights)
this_cov = np.dot(this_cov, centered_data.T)

pred_contact_mask = np.loadtxt("pred_reweight/beta_phi_240/pred_contact_mask.dat", dtype=bool)

this_cov_red = this_cov[pred_contact_mask][:,pred_contact_mask]
s = np.diag(np.sqrt(this_cov_red.diagonal()))
s_inv = np.linalg.inv(s)
this_corr_red = np.dot(np.dot(s_inv, this_cov_red), s_inv)

values, vectors = np.linalg.eig(this_corr_red)
sort_order = np.argsort(values)[::-1]

grp1 = vectors[sort_order[0]] > 0
grp2 = ~grp1

univ = MDAnalysis.Universe("pred_reweight/beta_phi_240/pred_contact.pdb")
univ.atoms.tempfactors = -2

univ.atoms[pred_contact_mask][grp2].tempfactors = 1
univ.atoms.write("blah.pdb")
