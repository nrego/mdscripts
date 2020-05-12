# Analyze results (in ml_tests) for model_fnot (or epsilon training)

import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os, glob, pathlib

def get_p_q(path):
    splits = path.parts[0].split('_')

    return int(splits[1]), int(splits[3])

## Run from sam_patterns directory...

fnames = sorted(glob.glob('P_*_Q_*/k_00/d_*/trial_0/PvN.dat'))
fnames 
# plus 3 for 6x6, 4x4, and 4x9 patches
n_files = len(fnames) + 3


p_q = np.zeros((n_files, 2))

energies_polar = np.zeros(n_files)
energies_nonpolar = np.zeros_like(energies_polar)

err_polar = np.zeros_like(energies_polar)
err_nonpolar = np.zeros_like(energies_polar)

states_polar = []
states_nonpolar = []

for i, fname in enumerate(fnames):

    path = pathlib.Path(fname)
    p, q = get_p_q(path)
    p_q[i] = p,q

    print('Extracting P: {:02d}  Q: {:02d}'.format(p,q))

    this_polar, this_err_polar = np.loadtxt(fname)[0,1:]

    n = p*q

    max_dir = 'k_{:02d}'.format(n)
    max_path = pathlib.Path(path.parts[0], max_dir, 'd_not', 'trial_0', 'PvN.dat')

    this_nonpolar, this_err_nonpolar = np.loadtxt(max_path)[0,1:]

    
    energies_polar[i] = this_polar
    energies_nonpolar[i] = this_nonpolar
    err_polar[i] = this_err_polar
    err_nonpolar[i] = this_err_nonpolar

    states_polar.append(State(np.array([], dtype=int), p, q))
    states_nonpolar.append(State(np.arange(n), p, q))

## 4 x 4 ###
ds = np.load('data/sam_pattern_04_04.npz')
energies = ds['energies']
err_energies = ds['err_energies']
states = ds['states']

min_idx = energies.argmin()
max_idx = energies.argmax()

# Make sure these are actually pure methyl/hydroxyl patches
assert states[min_idx].k_o == 0
assert states[max_idx].k_o == 16

idx = -3

p_q[idx] = 4, 4
energies_polar[idx] = energies[max_idx]
energies_nonpolar[idx] = energies[min_idx]
err_polar[idx] = err_energies[max_idx]
err_nonpolar[idx] = err_energies[min_idx]

states_polar.append(State(np.array([], dtype=int), 4, 4))
states_nonpolar.append(State(np.arange(16), 4, 4))

## 4 x 9 ###
ds = np.load('data/sam_pattern_04_09.npz')
energies = ds['energies']
err_energies = ds['err_energies']
states = ds['states']

min_idx = energies.argmin()
max_idx = energies.argmax()

# Make sure these are actually pure methyl/hydroxyl patches
assert states[min_idx].k_o == 0
assert states[max_idx].k_o == 36

idx = -2

p_q[idx] = 4, 9
energies_polar[idx] = energies[max_idx]
energies_nonpolar[idx] = energies[min_idx]
err_polar[idx] = err_energies[max_idx]
err_nonpolar[idx] = err_energies[min_idx]

states_polar.append(State(np.array([], dtype=int), 4, 9))
states_nonpolar.append(State(np.arange(36), 4, 9))

## 6 x 6 ###
ds = np.load('data/sam_pattern_06_06.npz')
energies = ds['energies']
err_energies = ds['err_energies']
states = ds['states']

min_idx = energies.argmin()
max_idx = energies.argmax()

# Make sure these are actually pure methyl/hydroxyl patches
assert states[min_idx].k_o == 0
assert states[max_idx].k_o == 36

idx = -1

p_q[idx] = 6, 6
energies_polar[idx] = energies[max_idx]
energies_nonpolar[idx] = energies[min_idx]
err_polar[idx] = err_energies[max_idx]
err_nonpolar[idx] = err_energies[min_idx]

states_polar.append(State(np.array([], dtype=int), 6, 6))
states_nonpolar.append(State(np.arange(36), 6, 6))


np.savez_compressed('pure_nonpolar', p=p_q[:,0], q=p_q[:,1], energies=energies_nonpolar, 
                    err_energies=err_nonpolar, states=states_nonpolar)

np.savez_compressed('pure_polar', p=p_q[:,0], q=p_q[:,1], energies=energies_polar, 
                    err_energies=err_polar, states=states_polar)




