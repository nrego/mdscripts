import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

from constants import k

import MDAnalysis

import argparse

import os, glob

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':30})
mpl.rcParams.update({'legend.fontsize':20})

fnames = glob.glob('*/pred/dewet_roc_dewet.dat')

sys_names = np.array([])

## For storing roc point (or its distance from ideal point (0,1)) at phi that performs best
min_dist_nofilter = np.array([])
min_dist_phob = np.array([])
min_dist_dewet = np.array([])

ideal_roc_nofilter = []
ideal_roc_phob = []
ideal_roc_dewet = []

for fname in fnames:
    subdir = os.path.dirname(fname)
    sys_name = os.path.dirname(subdir)

    sys_names = np.append(sys_names, sys_name)

    dist_nofilter = np.loadtxt('{}/dewet_dist_with_phi.dat'.format(subdir))
    dist_phob = np.loadtxt('{}/dewet_dist_with_phi_phob.dat'.format(subdir))
    dist_dewet = np.loadtxt('{}/dewet_dist_with_phi_dewet.dat'.format(subdir))

    roc_nofilter = np.loadtxt('{}/dewet_roc.dat'.format(subdir))
    roc_phob = np.loadtxt('{}/dewet_roc_phob.dat'.format(subdir))
    roc_dewet = np.loadtxt('{}/dewet_roc_dewet.dat'.format(subdir))

    idx_nofilter = np.argmin(dist_nofilter[:,-1])
    idx_phob = np.argmin(dist_phob[:,-1])
    idx_dewet = np.argmin(dist_dewet[:,-1])

    min_dist_nofilter = np.append(min_dist_nofilter, dist_nofilter[idx_nofilter, -1])
    min_dist_phob = np.append(min_dist_phob, dist_phob[idx_phob, -1])
    min_dist_dewet = np.append(min_dist_dewet, dist_dewet[idx_dewet, -1])

    ideal_roc_nofilter.append(roc_nofilter[idx_nofilter, 1:])
    ideal_roc_phob.append(roc_phob[idx_phob, 1:])
    ideal_roc_dewet.append(roc_dewet[idx_dewet, 1:])


ideal_roc_nofilter = np.array(ideal_roc_nofilter)
ideal_roc_phob = np.array(ideal_roc_phob)
ideal_roc_dewet = np.array(ideal_roc_dewet)

fig, ax = plt.subplots(figsize=(7.5,7.5))

ax.plot(ideal_roc_nofilter[:,0], ideal_roc_nofilter[:,1], 'ko', label='no filter')
ax.plot(ideal_roc_phob[:,0], ideal_roc_phob[:,1], 'bo', label='hydropathy filter')
ax.plot(ideal_roc_dewet[:,0], ideal_roc_dewet[:,1], 'ro', label='dewet filter')

ax.legend()
ax.set_xlim(0,1)
ax.set_xticks([0,0.5,1.0])
ax.set_ylim(0,1)
ax.set_yticks([0,0.5,1.0])

ax.plot(0, 1, '*')

fig.tight_layout()
fig.savefig('/Users/nickrego/Desktop/ideal_points.pdf', transparent=True)
fig.show()


from matplotlib import rc

fig, ax = plt.subplots(figsize=(18,7))

indices = np.arange(len(sys_names))

width = 0.25

ax.bar(indices, min_dist_nofilter, width=width, label='no filter', color='k')
ax.bar(indices+width, min_dist_phob, width=width, label='hydropathy filter', color='b')
ax.bar(indices+(2*width), min_dist_dewet, width=width, label='dewet filter', color='r')

#ax.legend(loc=0)
ax.set_ylabel(r'd')
ax.set_xticks(indices+width)
ax.set_xlim(-0.5, len(sys_names))
rc('text', usetex=False)
rc('xtick', labelsize=20)
ax.set_xticklabels(sys_names, rotation='vertical')

#ax.set_ylim(10,20)
fig.tight_layout()
#fig.savefig('/Users/nickrego/Desktop/legend.pdf', transparent=True)
fig.savefig('/Users/nickrego/Desktop/min_dist.pdf', transparent=True)
