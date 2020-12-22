
import numpy as np

import argparse
import logging

import scipy

import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed
import math

import sys

from whamutils import get_negloghist, extract_and_reweight_data

## MAKE 2d F_v^phi(N) plot in N, phi

def plot_errorbar(bb, dat, err, **kwargs):
    plt.plot(bb, dat, **kwargs)
    plt.fill_between(bb, dat-err, dat+err, alpha=0.5)

print('Constructing Nv v phi, chi v phi...')
sys.stdout.flush()
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
bins = np.arange(0, max_val+1, 1)

## In kT!
beta_phi_vals = np.arange(0,6.02,0.02)

print('WHAMing and reweighting to phi-ens')
sys.stdout.flush()

# Make sure our PV(N) goes to high enough N
maxn = np.ceil(max(all_data.max(), all_data_N.max()))
bins = np.arange(maxn+3)

all_neglogpdist, all_neglogpdist_N, all_avg, all_chi, all_avg_cube, all_chi_N, all_cov_cube = extract_and_reweight_data(all_logweights, all_data, all_data_N, bins, beta_phi_vals)

## Find 2d FvN with phi
#.    shape: (n_phi_vals, n_N_vals)
fvn_2d = np.zeros((beta_phi_vals.size, all_neglogpdist.size))

for i, beta_phi_val in enumerate(beta_phi_vals):
    if i % 100 == 0:
        print("bphi: {:.2f}".format(beta_phi_val))

    bias_logweights = all_logweights - beta_phi_val*all_data
    bias_logweights -= bias_logweights.max()

    this_neglogpdist = get_negloghist(all_data, bins, bias_logweights)
    fvn_2d[i] = this_neglogpdist

nn, phiphi = np.meshgrid(bins[:-1], beta_phi_vals, indexing='ij')

max_e = 10
de = 2
cmap = 'jet'
norm = plt.Normalize(0, max_e)
levels = np.arange(0, max_e+de, de)

plt.close('all')
fig, ax = plt.subplots()
pc = ax.pcolormesh(nn, phiphi, fvn_2d.T, cmap=cmap, norm=norm, shading='gouraud')
cn = ax.contour(nn, phiphi, fvn_2d.T, levels=levels, colors='k')
cb = plt.colorbar(pc, ticks=levels)
cb.add_lines(cn)

np.savez_compressed("fvn_2d.dat", fvn_2d=fvn_2d.T, nn=nn, phiphi=phiphi, nvals=bins[:-1], beta_phi_vals=beta_phi_vals) 

