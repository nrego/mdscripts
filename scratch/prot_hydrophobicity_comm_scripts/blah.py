from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import glob, os

mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':30})

name_lup = {'1brs': 'barnase',
            '1ubq': 'ubiquitin',
            '1qgt': 'capsid',
            '1ycr': 'mdm2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate dehydrogenase'}

order = ['hydrophobin', 'capsid', 'lysozyme', 'mdm2', 'malate dehydrogenase', 'barnase']

from constants import k

beta = 1/(k*300)

order_idx = []
fnames = np.array([], dtype=str)
for key, val in name_lup.iteritems():
    if val in order:
        order_idx.append(order.index(val))
        fnames = np.append(fnames, '{}/phi_sims/ntwid_out.dat'.format(key))
fnames = fnames[np.argsort(order_idx)]

peak_analysis_dat = np.zeros((len(fnames), 6))

for i,f in enumerate(fnames):
    dat = np.loadtxt(f)

    avg_0 = dat[0,1]
    var_0 = dat[0,2]

    peak_idx = np.argmax(dat[:,2])
    peak_val = dat[peak_idx,2]

    peak_analysis_dat[i,...] = avg_0, var_0, peak_val, peak_val/avg_0, peak_val/var_0, peak_val/avg_0**2

