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

# Find all beta phi vals at which this rho crosses threshold s
def interp_cross(beta_phi_vals, this_rho, s=0.5):
    under_mask = this_rho < s
    cross_mask = np.append(np.diff(under_mask), False)

    if cross_mask.sum() == 0:
        return np.array([beta_phi_vals[-1]])

    cross_bphis = []
    cross_indices = np.arange(beta_phi_vals.size)[cross_mask]

    for i_cross in cross_indices:
        if this_rho[i_cross] >= s:
            assert this_rho[i_cross+1] < s
        else:
            assert this_rho[i_cross+1] >= s

        dbphi = np.round(beta_phi_vals[i_cross+1] - beta_phi_vals[i_cross], 4)
        slope = (this_rho[i_cross+1]-this_rho[i_cross]) / dbphi
        inter_dbphi = (s - this_rho[i_cross]) / slope

        cross_bphis.append(beta_phi_vals[i_cross] + inter_dbphi)

    return np.array(cross_bphis)


dat = np.load('phi_sims/ni_rad_weighted.dat.npz')
#dat = np.load('ni_weighted.dat.npz')
beta_phi_vals = dat['beta_phi']

# shape: (n_heavies, n_phi_vals)
avg_nis = dat['avg']
#smooth_avg = dat['smooth_avg']
assert avg_nis.shape[1] == beta_phi_vals.size

try:
    os.makedirs('reweight_data')
except:
    pass
os.chdir('reweight_data')

for i_phi, beta_phi in enumerate(beta_phi_vals):
    dirname = 'beta_phi_{:04g}'.format(int(np.round(beta_phi*1000)))
    try:
        os.makedirs(dirname)
    except:
        pass

    ni_phi = avg_nis[:, i_phi]
    #smooth_ni = smooth_avg[:, i_phi]

    #np.savez_compressed('{}/ni_reweighted.dat'.format(dirname), 
    #                    rho_water=ni_phi[None,:], smooth_rho_water=smooth_ni[None,:])

    np.savez_compressed('{}/ni_reweighted.dat'.format(dirname), rho_water=ni_phi[None,:])

## Save all rho data simultaneously
n0 = avg_nis[:,0]
rho_dat = avg_nis / n0[:,None]
n_atm = n0.size

# Array of bphi vals at which each atom crosses threshold (s=0.5)
cross_vals = np.zeros(n_atm, dtype=object)

for i_atm in range(n_atm):
    this_rho = rho_dat[i_atm]
    cross_vals[i_atm] = interp_cross(beta_phi_vals, this_rho, s=0.5)

## rho_cross contains all the bphi vals where rho_i<s
##   ODD number of crossings: atom dewetted at bphi=bphi_max
##   EVEN number of crossings: Atom wet at bphi=bphi_max
np.savez_compressed('rho_cross.dat', n0=n0, beta_phi=beta_phi_vals, rho_data=rho_dat, cross_vals=cross_vals)
