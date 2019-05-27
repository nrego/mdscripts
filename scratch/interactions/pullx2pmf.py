from __future__ import division, print_function

import numpy as np
from IPython import embed

import argparse
import os, glob

import re

splitter = lambda instr: instr.split('/')[0].split('_')[-1]

extractFloat = lambda string: map(float, re.findall(r"[-+]?\d*\.\d+|\d+", string))

parser = argparse.ArgumentParser("Convert pullx.xvg file to pmf.dat")
parser.add_argument('-f', metavar='INPUT', type=str,
                    help='pullx.xvg input file name')
parser.add_argument('--rstar', required=True, type=float,
                    help='Rstar for this pulling window (in nm)')
parser.add_argument('--kappa', default=1000.0, type=float,
                    help='Spring constant for this rstar (in kj/mol-nm^2), default: %s(default)')


args = parser.parse_args()

filename = args.f
kappa = args.kappa
rstar = args.rstar

dat = np.loadtxt(filename, comments=['@', '#'])

headerstr = 'kappa(kJ/mol-nm^2): {:.2f}\nrstar(nm): {:1.2f}\n\ntime (ps)  r (nm)'.format(kappa, rstar)

np.savetxt('pmf.dat', np.delete(dat, 1, axis=1), header=headerstr, fmt='%1.4f  %1.6f')

