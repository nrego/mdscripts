import os, glob
import numpy as np
from scratch.sam.util import *
import shutil

## Extracts all data from patterned dataset  ###

## Set this to taste ##
p = 3
q = 3

fnames = sorted(glob.glob('P_{:02d}_Q_{:02d}/*/d_*/trial_*/PvN.dat'.format(p,q)))

n_dat = len(fnames)

print("Extracting data for p={} q={}...".format(p,q))
print("...Found {} patterns".format(n_dat))

myfeat = np.zeros((n_dat, 3))
energies = np.zeros(n_dat)

energies_err = np.zeros_like(energies)

states = np.empty_like(energies, dtype=object)

print("\nExtracting data...\n")

for i, fname in enumerate(fnames):

    pvn = np.loadtxt(fname)[0]
    pt_idx = np.loadtxt('{}/this_pt.dat'.format(os.path.dirname(fname)), dtype=int)
    state = State(pt_idx, p, q)

    myfeat[i] = state.k_o, state.n_oo, state.n_oe
    energies[i] = pvn[1]
    energies_err[i] = pvn[2]
    states[i] = state

if not os.path.exists('data/'):
    os.makedirs('data/')

outdir = 'data/sam_pattern_{:02d}_{:02d}'.format(p, q)
print("\nDone. Write to {} dir".format(outdir))

np.savez_compressed(outdir, states=states, energies=energies, err_energies=energies_err) 


