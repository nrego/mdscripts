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

# Find rho_i, phi for each voxel i at each value phi from directory 'reweight_data'

beta_phis = np.load("phi_sims/ni_weighted.dat.npz")['beta_phi']

n_0 = np.load("reweight_data/beta_phi_000/ni_reweighted.dat.npz")['rho_water'][0]

rho_with_phi = np.empty((n_0.size, beta_phis.size))
rho_with_phi[:] = np.inf

for i, phi in enumerate(beta_phis):
    phival = int(np.round(phi*100))
    fname = "reweight_data/beta_phi_{:03d}/ni_reweighted.dat.npz".format(phival)

    n_phi = np.load(fname)['rho_water'][0]

    rho_phi = n_phi / n_0
    rho_with_phi[:, i] = rho_phi

mask = np.ma.masked_invalid(rho_with_phi).mask
rho_with_phi[mask] = np.inf
np.save('rho_with_phi.dat', rho_with_phi)


## Find when each voxel's beta phi_i^*
beta_phi_star = np.zeros(rho_with_phi.shape[0])

for i_vox in range(rho_with_phi.shape[0]):
    this_rho = rho_with_phi[i_vox]

    for i_phi in range(100,0,-1):
        if this_rho[i_phi] < 0.5:
            continue
        else:
            try:
                this_phi_star = beta_phis[i_phi+1]
            except IndexError:
                this_phi_star = np.inf
            beta_phi_star[i_vox] = this_phi_star
            break

