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
dat = np.load("ni_rad_weighted.dat.npz")

beta_phis = dat['beta_phi']
n_with_phi = dat['avg']
cov_with_phi = dat['cov']

smooth_avg = dat['smooth_avg']
smooth_cov = dat['smooth_cov']

n_0 = n_with_phi[:,0]

buried_mask = n_0 < 5

rho_with_phi = n_with_phi / n_0[:, None]


## Find when each voxel's beta phi_i^*
beta_phi_star = np.zeros(rho_with_phi.shape[0])
n_phi_vals = beta_phis.size

for i_vox in range(rho_with_phi.shape[0]):
    this_rho = rho_with_phi[i_vox]

    for i_phi in range(n_phi_vals-1,0,-1):
        if this_rho[i_phi] < 0.5:
            continue
        else:
            try:
                this_phi_star = beta_phis[i_phi+1]
            except IndexError:
                this_phi_star = 4
            beta_phi_star[i_vox] = this_phi_star
            break

np.savetxt("beta_phi_star_prot.dat", beta_phi_star)

