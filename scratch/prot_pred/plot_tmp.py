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

from constants import k

beta = 1 / (300*k)

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 30})

roc = np.loadtxt('roc.dat')
phob_roc = np.loadtxt('phob_roc.dat')
dewet_roc = np.loadtxt('dewet_roc.dat')

perf = np.loadtxt('dist_with_phi.dat')
phob_perf = np.loadtxt('phob_dist_with_phi.dat')
dewet_perf = np.loadtxt('dewet_dist_with_phi.dat')


fig, ax = plt.subplots(figsize=(7,5))
ax.plot(roc[:,1], roc[:,2], 'ko', markersize=8)
ax.plot(phob_roc[:,1], phob_roc[:,2], 'bo', markersize=8, label='hydropathy filter')
ax.plot(dewet_roc[:,1], dewet_roc[:,2], 'ro', markersize=8, label='Dewet filter')

fig.legend()
fig.savefig('/Users/nickrego/Desktop/roc_comparison.pdf')

fig, ax = plt.subplots(figsize=(7,5))
phi = perf[:,0] * beta
ax.plot(phi, perf[:,2], 'ko', markersize=8)
ax.plot(phi, phob_perf[:,2], 'bo', markersize=8, label='hydropathy filter')
ax.plot(phi, dewet_perf[:,2], 'ro', markersize=8, label='Dewet filter')

fig.legend()
fig.savefig('/Users/nickrego/Desktop/performance_comparison.pdf')


indices = np.arange(3)
fig, ax = plt.subplots(figsize=(7,5))
ax.bar(indices, [perf.min(), phob_perf.min(), dewet_perf.min()], width=0.7)

ax.set_xticks(indices)
ax.set_xticklabels(['no filter', 'phobic filter', 'dewet filter'])
for tick in ax.get_xticklabels():
    tick.set_rotation(45)


