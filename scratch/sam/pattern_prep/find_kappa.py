from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mdtools import dr
import numpy as np

import networkx as nx
from scipy.spatial import cKDTree

from sklearn import datasets, linear_model
from scipy.integrate import cumtrapz

from constants import k


parser = argparse.ArgumentParser('Extract kappa, delta n from equilibrium simulation')
parser.add_argument('-f', default='phiout.dat', type=str,
                    help='input file name (default: %(default)s)')
parser.add_argument('-b', default=500, type=int,
                    help='Start frame to cutoff (default: %(default)s)')
parser.add_argument('--alpha', default=3, type=float,   
                    help='Alpha val by which to choose kappa (default: %(default)s)')
parser.add_argument('--temp', default=300, type=float,
                    help='Operating temp for determining kT (default: %(default)s K)')

args = parser.parse_args()

kt = args.temp*k

ds = dr.loadPhi(args.f)
start = args.b
alpha = args.alpha

dat = ds.data[start:]['$\~N$']

mean = dat.mean()
var = dat.var()
std = dat.std()

print('\navg ntwid: {:0.2f}  var ntwid: {:0.2f}   (std: {:0.2f})'.format(mean, var, std))
print('  alpha: {:0.1f}'.format(alpha))

kappa = np.round((alpha*kt / var), 3)
dn = int(np.ceil(4*(np.sqrt(1+alpha) / alpha) * std))

max_n = np.ceil(mean + 2*std)
min_n = - np.ceil(mean / alpha + 2*std)


print('\n  kappa: {:.3f}  dn: {:d}'.format(kappa, dn))

nstar_pos = np.arange(0, max_n, dn)
nstar_neg = np.arange(min_n, 0, dn)
all_nstar = np.append(nstar_neg, nstar_pos)

print('\n  nstar vals: {} ({} vals)'.format(all_nstar, all_nstar.size))


