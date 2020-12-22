
import sys
import numpy as np
from math import sqrt
import argparse
import logging
import shutil
import MDAnalysis

from IPython import embed

from scipy.spatial import cKDTree
import itertools
#from skimage import measure

from rhoutils import rho, cartesian
from mdtools import ParallelTool

from constants import SEL_SPEC_HEAVIES, SEL_SPEC_HEAVIES_NOWALL
from mdtools.fieldwriter import RhoField
import sys
import argparse, os
from scipy.interpolate import interp2d

import matplotlib as mpl

from skimage import measure
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from whamutils import get_negloghist, extract_and_reweight_data

nvphi = np.loadtxt("old/n_v_phi.dat")
#nvphi = np.loadtxt("NvPhi.dat")
## Run from static_indus_systems/large_1ubq/sh_*
bphi_vals = np.arange(0, 3.1, 0.1)
#fvn = np.loadtxt("PvN.dat")
fvn = np.loadtxt("old/neglogpdist_N.dat")

mask = ~np.ma.masked_invalid(fvn[:,1]).mask
nvals = fvn[mask,0]
fvn = fvn[mask,1]
fvn -= fvn.min()
norm = np.exp(-fvn).sum()
fvn += np.log(norm)

z = np.polyfit(nvals, fvn, deg=10)
p = np.poly1d(z)
p_prime = np.polyder(p,1)
p_dprime = np.polyder(p,2)

fvn_fit = p(nvals)
norm = np.exp(-fvn_fit).sum()
fvn_fit += np.log(norm)

fvn_prime = p_prime(nvals)
fvn_dprime = p_dprime(nvals)

#plt.plot(fvn[mask,0], fvn_dprime)
avg_n = np.zeros_like(bphi_vals)
chi_n = np.zeros_like(bphi_vals)

for i, bphi in enumerate(bphi_vals):
    fvphin = fvn_fit + bphi*nvals
    fvphin -= fvphin.min()

    norm = np.exp(-fvphin).sum()
    fvphin += np.log(norm)

    this_avg_n = np.trapz(np.exp(-fvphin)*nvals, nvals)
    this_avg_n_sq = np.trapz(np.exp(-fvphin)*nvals**2, nvals)

    this_chi_n = this_avg_n_sq - this_avg_n**2

    avg_n[i] = this_avg_n
    chi_n[i] = this_chi_n

max_bphi = bphi_vals[np.argmax(chi_n)]

fvphistar = fvn_fit + max_bphi*nvals
fvphistar -= fvphistar.min()
norm = np.exp(-fvphistar).sum()
fvphistar += np.log(norm)

plt.close('all')
plt.plot(nvals, fvn)
plt.plot(nvals, fvn_fit, 'k--', label='fit')
plt.legend()
plt.show()

plt.close('all')
grad1 = np.gradient(fvn, nvals)
grad2 = np.gradient(grad1, nvals)
plt.plot(nvals, grad1)
plt.plot(nvals, fvn_prime, 'k--')
plt.legend()

plt.close('all')
plt.plot(nvals, fvn_dprime, 'k--')

plt.close('all')
plt.plot(nvphi[:,0], nvphi[:,1])
plt.plot(bphi_vals, avg_n, 'k--')

plt.close('all')
#plt.plot(nvphi[:,0], nvphi[:,2])
plt.plot(nvphi[:,0], np.loadtxt("old/var_n_v_phi.dat")[:,1])
plt.plot(bphi_vals, chi_n, 'k--')

plt.close('all')
plt.plot(nvals, fvphistar, 'k--')

