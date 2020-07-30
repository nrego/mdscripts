
import numpy as np

import argparse
import logging


import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed

import MDAnalysis

# Find point (xs) where y=ys from points with y below and above threshold
def interp1d(xlo, xhi, ylo, yhi, ys=0.5):

    m = (yhi - ylo) / (xhi - xlo)

    if m == 0:
        return xlo

    return xlo + (ys-ylo)/m


# Find rho_i, phi for each voxel i at each value phi from directory 'reweight_data'
dat = np.load("phi_sims/ni_rad_weighted.dat.npz")

beta_phi_vals = dat['beta_phi']
n_with_phi = dat['avg']
cov_with_phi = dat['cov']

#smooth_avg = dat['smooth_avg']
#smooth_cov = dat['smooth_cov']

n_0 = n_with_phi[:,0]
rho_with_phi = n_with_phi / n_0[:, None]
buried_mask = n_0 < 5


## Find each atom's beta phi_i^*
beta_phi_star = np.zeros(rho_with_phi.shape[0])
n_phi_vals = beta_phi_vals.size

beta_phi_star[:] = 4


for i_atm in range(rho_with_phi.shape[0]):
    this_rho = rho_with_phi[i_atm]

    ## Go down in bphi values until we go back over threshold
    for i_phi in range(n_phi_vals-1, -1, -1):
        
        if this_rho[i_phi] < 0.5:
            continue

        # We have gone over threshold
        else:
            # Find cross
            try:
                bphi_lo = beta_phi_vals[i_phi]
                bphi_hi = beta_phi_vals[i_phi+1]

                rho_lo = this_rho[i_phi]
                rho_hi = this_rho[i_phi+1]
                beta_phi_star[i_atm] = interp1d(bphi_lo, bphi_hi, rho_lo, rho_hi, 0.5)

                break
            # atom i still wet at bphi=4.0
            except IndexError:
                break

np.savetxt("beta_phi_star_prot.dat", beta_phi_star)

univ = MDAnalysis.Universe("bound/actual_contact.pdb")
assert univ.atoms.n_atoms == beta_phi_star.size
univ.atoms.tempfactors = beta_phi_star
univ.atoms[buried_mask].tempfactors = -2


univ.atoms.write("bphi.pdb")



