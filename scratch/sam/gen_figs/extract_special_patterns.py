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


mask_fnames = sorted(glob.glob('*.dat'))

names = []
states = []
energies = []
err_energies = []
feat_vec = []

for mask_fname in mask_fnames:
    mask = np.loadtxt(mask_fname, dtype=bool)
    dirname = mask_fname.split('.')[0]

    energy, err_energy = np.loadtxt('{}/PvN.dat'.format(dirname))[0, 1:]

    names.append(dirname)
    state = State(np.arange(36, dtype=int)[mask])
    states.append(state)

    feat_vec.append(np.array([state.k_o, state.n_oo, state.n_oe]))

    energies.append(energy)
    err_energies.append(err_energy)

names = np.array(names, dtype=object)
states = np.array(states, dtype=object)
energies = np.array(energies)
err_energies = np.array(err_energies)
feat_vec = np.array(feat_vec)

np.savez_compressed('sam_special_patterns.dat', names=names, states=states,
                    feat_vec=feat_vec, energies=energies, err_energies=err_energies)
