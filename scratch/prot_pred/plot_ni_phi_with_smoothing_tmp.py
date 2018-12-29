from __future__ import division, print_function

import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl

from mdtools import dr

from constants import k

import MDAnalysis

import argparse
from IPython import embed
import matplotlib as mpl
import os

homedir = os.environ['HOME']
mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

beta = 1/(300*k)

idx = 680
fnames = sorted(glob.glob('smooth_*_rho_pred/rho_i_with_phi.dat.npz'))

delta_beta_phis = np.array([float(fname.split('/')[0].split('_')[1]) for fname in fnames]) / 100.0

colors = cm.rainbow(np.linspace(0,1,delta_beta_phis.size))

fig, ax = plt.subplots(figsize=(8,7.5))
for i, fname in enumerate(fnames):

    subdir = os.path.dirname(fname)
    dat = np.load(fname)
    beta_phi_vals = dat['beta_phi']
    rho_i_phi = dat['rho_i']

    dat = np.load('{}/h_i_with_phi.dat.npz'.format(subdir))
    assert np.array_equal(beta_phi_vals, dat['beta_phi'])
    h_i_phi = dat['h_i']

    this_rho_i_phi = rho_i_phi[:,idx]
    thresh = this_rho_i_phi >= 0.5

    this_h_i_phi = h_i_phi[:,idx]


    ax.plot(beta_phi_vals, this_rho_i_phi, 'o-', color=colors[i], label=r'$\delta \beta \phi = {}$'.format(delta_beta_phis[i]))
    #plt.step(beta_phi_vals, thresh, '--', color=colors[i])

ax.legend()

ax.set_xlabel(r'$\beta \phi$')
#ax.set_ylabel(r'$\langle \sigma_{{{}}} \rangle_\phi$'.format(idx))
ax.set_ylabel(r'$\langle \rho_{{{}}} \rangle_\phi$'.format(idx))


fig.tight_layout()
#plt.show()
fig.savefig('{}/Desktop/smooth_rho_{}.png'.format(homedir, idx))
plt.close('all')

