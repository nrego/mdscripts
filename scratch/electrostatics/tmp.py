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

k_e = 138.9354859 * 10.0
univ = MDAnalysis.Universe('run.tpr', 'traj.xtc')


rvals = np.zeros(univ.trajectory.n_frames)
atm1 = univ.atoms[0]
atm2 = univ.atoms[1]

for i, ts in enumerate(univ.trajectory):
    rvals[i] = np.linalg.norm( atm1.position - atm2.position )

tab_dat = np.loadtxt('coul.xvg')
plt.plot(tab_dat[:,0]*10, tab_dat[:,1]/10)

e_dat = np.loadtxt('test_tab/energy.xvg', comments=['@', '#'])

pot_sr = (e_dat[:,1]+e_dat[:,2]) / k_e

plt.scatter(rvals, -pot_sr)
plt.show()