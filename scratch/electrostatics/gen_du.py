from __future__ import division, print_function

import numpy as np
from IPython import embed

import argparse
import os, glob

splitter = lambda instr: instr.split('/')[0].split('_')[-1]

parser = argparse.ArgumentParser("Combine results of g_energy on trajectory evaluated with different topologies into single file")
parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                    help='Input file names')
parser.add_argument('--this-lambda', type=float, required=True,
                    help='Lambda of this value')

args = parser.parse_args()

this_lambda = args.this_lambda

infiles = sorted(args.input)

lam_vals = np.array([float(splitter(infile))/10.0 for infile in infiles])

u0 = np.loadtxt(infiles[0], comments=['@','#'])
timepts = u0[:,0]


energies = np.zeros((timepts.size, lam_vals.size))


for i, infile in enumerate(infiles):
    u_lam = np.loadtxt(infile, comments=['@','#'])
    energies[:,i] = u_lam[:,1]
    assert np.array_equal(u_lam[:,0], timepts)

energies -= energies[:,0][:,None]

for_lam_str = ' '.join(['{:0.2f}'.format(lam) for lam in lam_vals])

header = 'this_lmbda: {:0.2f}\nlmbdas: {}\ntime (ps) U_lmbda-U_0'.format(this_lambda, for_lam_str)

np.savetxt('du.dat', np.hstack((timepts[:,None], energies)), fmt='%0.6f', header=header)


