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

import numpy as np
import sympy

from scipy.special import binom

import itertools

from scratch.sam.util import *

import pickle

import shutil



## Re-initialize states for sam_pattern_[] dataset
fnames = ['data/sam_pattern_06_06.npz', 'data/sam_pattern_04_09.npz', 'data/sam_pattern_04_04.npz', 'data/sam_pattern_02_02.npz']


for fname in fnames:
    print("\nfreshening states for: {}".format(fname))
    print("##################\n")
    ds = np.load(fname)

    try:
        old_states = ds['states']
    except KeyError:
        print("no states. Exiting.")

    new_states = np.empty_like(old_states)

    for i, state in enumerate(old_states):

        if i % 100 == 0:
            print("doing state {} out of {}".format(i+1, new_states.shape[0]))
        try:
            p = state.ny
        except AttributeError:
            p = state.P

        try:
            q = state.nz
        except AttributeError:
            q = state.Q

        new_state = State(state.pt_idx, p=p, q=q)

        new_states[i] = new_state

    new_dict = dict()
    new_dict['states'] = new_states

    for k, v in ds.items():
        if k == 'states':
            continue
        new_dict[k] = v

    np.savez_compressed(fname, **new_dict)

    print("...Done")



