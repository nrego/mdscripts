from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

import warnings

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import networkx as nx
from scipy.spatial import cKDTree
from scipy.special import erf, erfc, erfcinv

from scipy.integrate import cumtrapz

import math
import itertools

# Electrostatic constant, in units: kJ mol^-1 A e^-2
k_e = 138.9354859

# Between gaussian and point charge, or two gaussians
def coul_erf(qi, qj, rij, alpha):
    zero_idx = rij == 0

    lim_zero = 2*alpha/np.sqrt(np.pi)
    vals = (qi*qj)/(rij) * erf(alpha*rij)
    vals[zero_idx] = lim_zero

    fprime = (1/rij)*( ((2*alpha)/np.sqrt(np.pi)) * np.exp(-alpha*alpha*rij*rij) - vals )
    fprime[0] = -0

    return vals, fprime


parser = argparse.ArgumentParser('Generate table for gaussian charges')
parser.add_argument('--sigma', default=0.05, type=float,
                    help='Width of gaussian, in nm (default: %(default)s)')
parser.add_argument('--dr', default=5e-4, type=float,
                    help='Spacing for r points, in nm (default: %(default)s)')
parser.add_argument('--do-plot', action='store_true',
                    help='Plot the gaussian charge (default: do not plot)')
parser.add_argument('--rtol', type=float, default=1e-5,
                    help='rtol gromacs parameter for determining ewald screen width (default: %(default)s)')
parser.add_argument('--rcut', type=float, default=1.0,
                    help='SR cutoff, in nm; used with \'rtol\' for determining ewald screen width (default: %(default)s)')
args = parser.parse_args()

do_plot = args.do_plot
dr = args.dr
sig = args.sigma

rvals = np.arange(0, 3+dr, dr)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    one_over_r = 1/rvals

excl_idx = one_over_r > 25
one_over_r[excl_idx] = 0
one_over_rsq = one_over_r**2

# units: nm
alpha = 1/(np.sqrt(2)*sig)

# Gaus-point potential 
# (1/r)*erf(alpha r); units: nm
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vals_gaus_pt, prime_gaus_pt = coul_erf(1, 1, rvals, alpha)

# Gaus-gaus potential
# (1/r)*erf((alpha/sqrt(2)) r)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    vals_gaus_gaus, prime_gaus_gaus = coul_erf(1, 1, rvals, alpha/np.sqrt(2))

## for Ewald screening gaussian - check that it's wider than 
#     gaussian test charge
screen_alpha = erfcinv(1e-5)
# in nm
screen_sig = 1/(np.sqrt(2)*screen_alpha)
screen, screen_prime = coul_erf(1, 1, rvals, screen_alpha)

assert sig <= screen_sig
print("Test charge sigma (nm): {:0.4f}".format(sig))
print("Ewald screening charge sigma: {:0.4f} (nm)".format(screen_sig))

if do_plot:
    plt.plot(rvals, one_over_r, label='1/r')
    plt.plot(rvals, vals_gaus_pt, 'k--', label='gaus')
    plt.plot(rvals, screen, label='screen')

    plt.legend()
    plt.ylim(0,2)
    plt.show()


    # plt.plot(rvals, one_over_r, label='1/r')
    # plt.plot(rvals, one_over_r-screen, label='screened')
    # plt.legend()
    # plt.ylim(0,2)
    # plt.show()


    # plt.plot(rvals, vals_gaus_pt, label='gaus')
    # plt.plot(rvals, vals_gaus_pt-screen, label='screened')
    # plt.legend()
    # plt.show()

    # plt.plot(rvals, one_over_r-screen, label='1/r screened')
    # plt.plot(rvals, vals_gaus_pt-screen, 'k--', label='gaus screened')
    # plt.legend()
    # plt.ylim(0,2)
    # plt.show()


    # Test derivatives
    plt.plot(rvals, k_e*prime_gaus_pt)
    plt.plot(rvals, k_e*np.gradient(vals_gaus_pt, np.diff(rvals)[0]), 'k--')

    plt.show()


n_vals = rvals.size

# General (pt-pt charge interaction, i.e. 1/r) table
force = one_over_rsq
if force[0] == 0:
    force[0] = 0

outarray = np.vstack([rvals, one_over_r, force, np.ones(n_vals), np.zeros(n_vals), np.ones(n_vals), np.zeros(n_vals)])

header = 'r(nm)      coulombic   coul_force   vdw...(not used)'
np.savetxt('table.xvg', outarray.T, fmt='%1.12e ', header=header)

# (pt charge-gaus) table
force = -prime_gaus_pt
if force[0] == 0:
    force[0] = 0

outarray = np.vstack([rvals, vals_gaus_pt, force, np.ones(n_vals), np.zeros(n_vals), np.ones(n_vals), np.zeros(n_vals)])

header = 'r(nm)      coulombic   coul_force   vdw...(not used)'
np.savetxt('table_G_Pt.xvg', outarray.T, fmt='%1.12e ', header=header)


# gaus-gaus table
force = -prime_gaus_gaus
if force[0] == 0:
    force[0] = 0

outarray = np.vstack([rvals, vals_gaus_gaus, force, np.ones(n_vals), np.zeros(n_vals), np.ones(n_vals), np.zeros(n_vals)])
np.savetxt('table_G_G.xvg', outarray.T, fmt='%1.12e ', header=header)
