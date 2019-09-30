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
    dirname = 'beta_phi_{:03g}'.format(int(np.round(beta_phi*100)))
    try:
        os.makedirs(dirname)
    except:
        pass

    ni_phi = avg_nis[:, i_phi]
    #smooth_ni = smooth_avg[:, i_phi]

    #np.savez_compressed('{}/ni_reweighted.dat'.format(dirname), 
    #                    rho_water=ni_phi[None,:], smooth_rho_water=smooth_ni[None,:])

    np.savez_compressed('{}/ni_reweighted.dat'.format(dirname), rho_water=ni_phi[None,:])
