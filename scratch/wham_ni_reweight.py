from __future__ import division, print_function

import numpy as np

import argparse
import logging

import MDAnalysis
import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython import embed

from constants import k

from scratch.wham_reweight import get_negloghist, extract_and_reweight_data

## Load in sim data and WHAM weights ##
all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_N']

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)

bins = np.arange(0, max_val+1, 1)

### Now input all <n_i>_\phi's for a given i ###
print('')
print('Extracting all n_i\'s...')
beta_phis = np.linspace(0,4,1001)


n_heavies = None


n_i_dat_fnames = sorted(glob.glob('phi_*/rho_data_dump_rad_6.0.dat.npz'))
#n_i_dat_fnames[0] = 'equil/rho_data_dump_rad_6.0.dat.npz'

# Ntwid vals, only taken every 1 ps from each window
all_data_reduced = np.array([])
all_logweights_reduced = np.array([])

# Will have shape (n_heavies, n_tot)
all_data_n_i = None
n_files = len(n_i_dat_fnames)

phiout_0 = np.loadtxt('phi_000/phiout.dat')

start_idx = 0
# number of points in each window
shift = all_data.shape[0] // n_files
assert all_data.shape[0] % n_files == 0
## Gather n_i data from each umbrella window (phi value)
for i in range(n_files):
    ## Need to grab every 10th data point (N_v, and weight)
    this_slice = slice(start_idx, start_idx+shift, 100)
    start_idx += shift
    new_data_subslice = all_data[this_slice]
    new_weight_subslice = all_logweights[this_slice]

    all_data_reduced = np.append(all_data_reduced, new_data_subslice)
    all_logweights_reduced = np.append(all_logweights_reduced, new_weight_subslice)
    
    fname = n_i_dat_fnames[i]

    ## Shape: (n_heavies, n_frames) ##
    # Cut off first 30 (300 ps) since we started this at 200, while
    #   WHAM stuff starts at 500 ps
    n_i = np.load(fname)['rho_water'][30:,:].T
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

print('WHAMing each n_i...')

# Shape: (n_atoms, beta_phis.shape)
avg_nis = np.zeros((n_heavies, beta_phis.size))
chi_nis = np.zeros((n_heavies, beta_phis.size))
for i_atm in range(n_heavies):

    if i_atm % 100 == 0:
        print('  i: {}'.format(i_atm))
    neglogpdist, neglogpdist_ni, beta_phis, _, _, avg_ni, chi_ni = extract_and_reweight_data(all_logweights_reduced, all_data_reduced, all_data_n_i[i_atm], bins, beta_phis)

    avg_nis[i_atm, :] = avg_ni
    chi_nis[i_atm, :] = chi_ni

np.savez_compressed('ni_weighted.dat', avg=avg_nis, var=chi_nis, beta_phi=beta_phis)


## Now find the beta phi_i stars - when each subvolume dewets for good 
# Shape: (36, n_frames)
rho_nis = avg_nis / avg_nis[:, 0][:,None]

beta_phi_stars = np.zeros(rho_nis.shape[0])
beta_phi_stars[:] = beta_phis[-1]

for i_vol in range(rho_nis.shape[0]):
    this_rho = rho_nis[i_vol]

    for i_phi in range(100,0,-1):
        if this_rho[i_phi] < 0.5:
            continue
        else:
            try:
                this_phi_star = beta_phis[i_phi+1]
            # Not dewet at highest phi - break out
            except IndexError:
                break
            beta_phi_stars[i_vol] = this_phi_star
            break


## Now map the beta phi stars, etc to the actual patch indexing scheme

univ = MDAnalysis.Universe('struct.gro')
res = univ.residues[-36:]

start_idx = univ.residues[-37].resid + 1

mapping = res.resids - start_idx

mapped_beta_phi_star = np.zeros_like(beta_phi_stars)
mapped_beta_phi_star[mapping] = beta_phi_stars

np.savetxt("beta_phi_i_star.dat", mapped_beta_phi_star, fmt='%1.2f')
