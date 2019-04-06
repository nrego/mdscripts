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

fnames = glob.glob('*/*/contact_atom_charges.dat')


for fname in fnames:
    headdir = os.path.dirname(os.path.dirname(fname))

    print("PROT: {}".format(headdir))
    contact_charge = np.loadtxt(fname)
    n_atms = contact_charge.size

    f_k_bulk = np.loadtxt('{}/bulk/f_k_all.dat'.format(headdir))[-1]
    f_k_bound = np.loadtxt('{}/bound/f_k_all.dat'.format(headdir))[-1]

    dg = f_k_bulk - f_k_bound

    print("  n_inter_atoms: {}".format(n_atms))
    print("  dg_coul: {:0.1f}".format(dg))
    print("  dg per atom: {:0.4f}".format(dg/n_atms))