import numpy as np

import argparse
import logging

import scipy

import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed
import math

import sys

from whamutils import get_negloghist, extract_and_reweight_data

# Reweight a dataset at a bphi ensemble
def reweight_ds(logweights, bphi, ntwid, data):
    bias_logweights = logweights - beta_phi_val*ntwid
    bias_logweights -= bias_logweights.max()
    norm = np.log(np.sum(np.exp(bias_logweights)))
    bias_logweights -= norm

    bias_weights = np.exp(bias_logweights)

## Find covariance matrix for <rho_i rho_j> at a given phi value ##

bphi = 2.20 
old_bmask = np.loadtxt('../../pred_reweight/beta_phi_000/buried_mask.dat', dtype=bool)
old_cmask = np.loadtxt('../../pred_reweight/beta_phi_{:03d}/pred_contact_mask.dat'.format(int(np.round(bphi*100))), dtype=bool)
assert old_cmask[old_bmask].sum() == 0

print('Extracting logweights for each observation')
sys.stdout.flush()
temp = 300

beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})


### EXTRACT DATA FROM WHAM ###
  # logweights for each sim; used for reweighting whatever we please #

all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_ntwid = all_data_ds['data']

### Now input all n_i_\phi's for each window (will correspond to logweights, for each atom i) ###
print('')
print('Extracting all n_i\'s...')
sys.stdout.flush()

n_heavies = None

n_i_dat_fnames = np.append(sorted(glob.glob('phi*/rho_data_dump_rad_6.0.dat.npz')), sorted(glob.glob('nstar*/rho_data_dump_rad_6.0.dat.npz')))
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


# Now get normalizing factors <n_i>0 for each atom (which we've already calculated)
ds = np.load('ni_rad_weighted.dat.npz')
n0 = ds['avg'][:,0]

buried_mask = n0 < 5
assert np.array_equal(buried_mask, old_bmask)


