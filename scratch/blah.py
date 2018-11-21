from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
import logging

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

import MDAnalysis

import os, glob
import cPickle as pickle


mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 30})

fnames = glob.glob('d_*/trial_0/PvN.dat')

with open('pt_idx_data.pkl', 'r') as fin:
    rms_bins, occupied_idx, positions, sampled_pt_idx = pickle.load(fin)

bb = rms_bins[:-1]
mus = np.zeros_like(bb)
mus[:] = np.nan
errs = np.zeros_like(bb)

for fname in fnames:
    dirname = os.path.dirname(os.path.dirname(fname))
    d = float(dirname.split('_')[-1]) / 100.0

    bin_idx = np.digitize(d, rms_bins) - 1

    dat = np.loadtxt(fname)
    dg = dat[0, 1]
    err = dat[0, 2]
    mus[bin_idx] = dg
    errs[bin_idx] = err