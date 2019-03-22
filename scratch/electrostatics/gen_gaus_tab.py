from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

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

do_plot = True

# Between gaussian and point charge, or two gaussians
def coul_erf(qi, qj, rij, alpha):
    zero_idx = rij == 0

    lim_zero = 2*alpha/np.sqrt(np.pi)
    vals = (qi*qj)/(rij) * erf(alpha*rij)
    vals[zero_idx] = lim_zero

    fprime = (1/rij)*( ((2*alpha)/np.sqrt(np.pi)) * np.exp(-alpha*alpha*rij*rij) - vals )
    fprime[0] = -0

    return vals, fprime


dr = 5e-4
rvals = np.arange(0, 30+dr, dr)

one_over_r = 1/rvals

# units: nm
sig = 0.05
alpha = 1/(np.sqrt(2)*sig)

vals, prime = coul_erf(1, 1, rvals, alpha)

screen_alpha = erfcinv(1e-5)
# in nm
screen_sig = 1/(np.sqrt(2)*screen_alpha)
screen, screen_prime = coul_erf(1, 1, rvals, screen_alpha)

if do_plot:
    plt.plot(rvals, one_over_r, label='1/r')
    plt.plot(rvals, vals, 'k--', label='gaus')
    plt.plot(rvals, screen, label='screen')

    plt.legend()
    plt.ylim(0,2)
    plt.show()


    plt.plot(rvals, one_over_r, label='1/r')
    plt.plot(rvals, one_over_r-screen, label='screened')
    plt.legend()
    plt.ylim(0,2)
    plt.show()


    plt.plot(rvals, vals, label='gaus')
    plt.plot(rvals, vals-screen, label='screened')
    plt.legend()
    plt.show()

    plt.plot(rvals, one_over_r-screen, label='1/r screened')
    plt.plot(rvals, vals-screen, 'k--', label='gaus screened')
    plt.legend()
    plt.ylim(0,2)
    plt.show()


    # Test derivatives
    plt.plot(rvals, prime)
    plt.plot(rvals, np.gradient(vals, np.diff(rvals)[0]), 'k--')

    plt.show()


force = -prime
if force[0] == 0:
    force[0] = 0

n_vals = rvals.size
outarray = np.vstack([rvals, vals, force, np.ones(n_vals), np.zeros(n_vals), np.ones(n_vals), np.zeros(n_vals)])



header = 'r(nm)      coulombic   coul_force   vdw...(not used)'
np.savetxt('gaus.xvg', outarray.T, fmt='%1.12e ', header=header)
