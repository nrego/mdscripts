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

from whamutils import get_negloghist, extract_and_reweight_data

import glob


## Compare average N for each umbrella window to averages near beta phi* ##

def get_close(all_avg, avg): 
    diff = all_avg - avg 
    return np.abs(diff).argmin()

temp = 300

beta = 1./(k*temp)
dtype = np.float64

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize': 50})


### EXTRACT MBAR DATA ###

dat = np.loadtxt("NvPhi.dat")

beta_phi_vals, all_avg, all_chi = [ arr.squeeze() for arr in np.split(dat, 3, axis=1) ]

max_idx = np.argmax(all_chi)
print("Beta phi star: {:.4f}".format(beta_phi_vals[max_idx]))
print(" Average: {:.2f}".format(all_avg[max_idx]))

## Extract Nstar averages
fnames = sorted(glob.glob("Nstar_*long/ntw*dat"))

avgs = []
for fname in fnames:
    ds = dr.loadPhi(fname)
    avg = ds.data[500:]['$\~N$'].mean()
    print("Nstar: {:.2f}    avg: {:.2f}".format(ds.Nstar, avg))
    avgs.append(avg)

indices = [ get_close(all_avg, avg) for avg in avgs ]

plt.close('all')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(beta_phi_vals, all_avg)
ax2.plot(beta_phi_vals, all_chi)
ax2.plot(beta_phi_vals[indices], all_chi[indices], 'yD')




