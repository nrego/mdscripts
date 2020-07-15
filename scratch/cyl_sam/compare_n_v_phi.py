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

all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_aux']

boot_indices = np.load('boot_indices.dat.npy')
dat = np.load('boot_fn_payload.dat.npy')
n_iter = boot_indices.shape[0]

max_val = int(np.ceil(np.max((all_data, all_data_N))) + 1)
bins = np.arange(0, max_val+1, 1).astype(int)

## In kT!
beta_phi_vals = np.arange(0,6.02,0.02)

all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_N, all_chi_N, all_cov_N = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)

## Extract Nstar averages
fnames = sorted(glob.glob("Nstar_*/ntw*dat"))

avgs = []
for fname in fnames:
    ds = dr.loadPhi(fname)
    avg = ds.data[500:]['$\~N$'].mean()
    print("Nstar: {:.2f}    avg: {:.2f}".format(ds.Nstar, avg))
    avgs.append(avg)

end_idx = 6
avgs = np.array(avgs)[np.array([0,1,2,8,9,10])]

indices = [ get_close(all_avg, avg) for avg in avgs ]

plt.close('all')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(beta_phi_vals, all_avg)
ax2.plot(beta_phi_vals, all_chi)
ax2.plot(beta_phi_vals[indices], all_chi[indices], 'yD')




