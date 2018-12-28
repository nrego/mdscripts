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

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

beta = 1/(300*k)

idx = 604
fnames = sorted(glob.glob('smooth_*/rho_i_with_phi.dat.npz'))

delta_beta_phis = np.array([float(fname.split('/')[0].split('_')[-2]) for fname in fnames]) / 100.0

colors = cm.rainbow(np.linspace(0,1,delta_beta_phis.size))

fname = 'smooth_000_pred/rho_i_with_phi.dat.npz'
dat = np.load(fname)
beta_phi_vals = dat['beta_phi']
rho_i_phi = dat['rho_i']
this_rho_i_phi = rho_i_phi[:,idx]
thresh = this_rho_i_phi >= 0.5

plt.plot(beta_phi_vals, this_rho_i_phi, 'o', color=colors[0])
plt.step(beta_phi_vals, thresh, '-', where='post', color=colors[0])

plt.legend()

plt.xlabel(r'$\beta \phi$')
plt.ylabel(r'$\langle \rho_{{{}}} \rangle_\phi < 0.5$'.format(idx))

plt.tight_layout()
plt.show()
plt.savefig('/home/nick/Desktop/smooth_thresh_i_{}.png'.format(idx))
plt.close('all')