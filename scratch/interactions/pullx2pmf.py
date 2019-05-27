from __future__ import division, print_function

import numpy as np
from IPython import embed

import argparse
import os, glob

import re

splitter = lambda instr: instr.split('/')[0].split('_')[-1]

extractFloat = lambda string: map(float, re.findall(r"[-+]?\d*\.\d+|\d+", string))

parser = argparse.ArgumentParser("Convert dhdl.xvg file to du.dat")
parser.add_argument('-f', metavar='INPUT', type=str,
                    help='dhdl nput file name')


args = parser.parse_args()

filename = args.f

lam_vals = []
with open(filename, 'r') as f:
    
    for line in f:
        #print(line)
        if line.startswith('#'):
            continue
        elif line.startswith('@'):
            
            if line.find('T =') != -1:
                this_lambda = extractFloat(line)[1]
            elif re.findall('s[0-9]+', line):
                lam_vals.append(extractFloat(line)[1])
        else:
            break

dat = np.loadtxt(filename, comments=["@", "#"])
timepts = dat[:,0]
energies = dat[:,1:]
energies -= energies[:,0][:,None]

print("This lambda: {:0.2f}".format(this_lambda))

for_lam_str = ' '.join(['{:0.2f}'.format(lam) for lam in lam_vals])

header = 'this_lmbda: {:0.2f}\nlmbdas: {}\ntime (ps) U_lmbda-U_0'.format(this_lambda, for_lam_str)

np.savetxt('du.dat', np.hstack((timepts[:,None], energies)), fmt='%0.6f', header=header)


