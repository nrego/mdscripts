from __future__ import division, print_function

import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

from constants import k

import MDAnalysis

import argparse

from scipy.spatial import cKDTree


ds = dr.loadPhi('phiout.dat')
dat = ds.data[500:]['$\~N$']
mean = dat.mean()
var = dat.var(ddof=1)
std = dat.std(ddof=1)

kt = 300 * k
alpha = 3

kappa = alpha*kt/var
dn = int(np.ceil(4 * (np.sqrt(1+alpha)/alpha) * std))

neg_nstar = mean / alpha

print('Mean: {:0.2f}  var: {:0.2f}  std: {:0.2f}'.format(mean, var, std))
print('    kappa: {:0.4f}  dn: {:02d}  neg_nstar: {:0.2f}'.format(kappa, dn, neg_nstar))


