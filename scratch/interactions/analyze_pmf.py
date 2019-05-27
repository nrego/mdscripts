from __future__ import division, print_function

import numpy as np
from IPython import embed

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt

import argparse
import os, glob

import sys

from mdtools import dr

from constants import k

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 30})
mpl.rcParams.update({'text.usetex': False})


splitter = lambda instr: instr.split('/')[0].split('_')[-1]

parser = argparse.ArgumentParser("Analyze PMF distances from pull code in pmf.dat")
parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                    help='Input file names')
parser.add_argument('-b', '--start', default=0, type=float,
                    help='start time point')
parser.add_argument('--plotDistAll', action='store_true',
                    help='If true, plot all rdist distributions')
parser.add_argument('--bin-width', type=float, default=0.001,
                    help='Binning width for histograms, in nm - default: %(default)s')

args = parser.parse_args()

infiles = sorted(args.input)

dat_min = np.inf
dat_max = -np.inf

n_files = len(infiles)
cmap = cm.tab20c
norm = plt.Normalize(0, n_files-1)

for fname in infiles:
    ds = dr.loadPMF(fname)

    if np.array(ds.data).max() > dat_max:
        dat_max = np.array(ds.data).max()
    if np.array(ds.data).min() < dat_min:
        dat_min = np.array(ds.data).min()

if len(infiles) == 1:
    #embed()
    plt.plot(ds.data)
    plt.show()
    sys.exit()

dat_max = np.ceil(dat_max)
dat_min = np.floor(dat_min)

start_time = args.start

rbins = np.arange(dat_min, dat_max+args.bin_width, args.bin_width)

if args.plotDistAll:
    for i, ds in enumerate(dr.datasets.values()):
        color = cmap(norm(i))
        dat = np.array(ds.data[start_time:])
        
        hist, bb = np.histogram(dat, bins=rbins)

        plt.plot(bb[:-1], hist, label='{}'.format(ds.title), color=color)

    plt.legend()
    plt.show()





