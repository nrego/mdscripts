
import numpy as np
import pandas as pd
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr, get_object
import scipy.integrate
#from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
from scipy.optimize import minimize
import pymbar
import time

from mdtools import ParallelTool

from constants import k

from whamutils import kappa, grad_kappa, gen_data_logweights, WHAMDataExtractor, callbackF

import matplotlib as mpl

import matplotlib.pyplot as plt

import itertools
from scipy.special import binom

from scratch.sam.util import *
from scratch.sam.enumerate_design import GetDelta
import os, glob


parser = argparse.ArgumentParser('run greedy monomer design')
parser.add_argument('-p', default=6)
parser.add_argument('-q', default=6)
parser.add_argument('--do-plot', action='store_true')
parser.add_argument('--n-trials', default=1000)
parser.add_argument('--idx', default=0)
parser.add_argument('--reg-file', default='data/sam_reg_coef.npy')

args = parser.parse_args()

homedir = os.environ['HOME']

do_plot = args.do_plot
n_trials = args.n_trials

p = args.p
q = args.q
n = p*q

idx = args.idx

### Run a trajectory to break and build hydrophobicity (placing monomers) ###
#############################################################################
## (Run from sam_data directory) ##
np.random.seed()


reg = np.load(args.reg_file).item()

alpha1, alpha2, alpha3 = reg.coef_

alpha_n_cc = alpha2
alpha_n_ce = alpha2 - alpha3

state = State(np.arange(p*q), p=p, q=q)

delta = GetDelta(state.adj_mat, state.ext_count, alpha_n_cc, alpha_n_ce)

indices = np.arange(n)

## Break phobicity: Add hydroxyls to pure phobic ##
###################################################
print("Breaking phobic...\n")
sys.stdout.flush()

# A of visited states for each ko
break_state_count = np.ones((n_trials, n+1, n), dtype=int)

for i_trial in range(n_trials):
    print("trial: {}".format(i_trial))

    x0 = np.ones(n, dtype=bool)

    if do_plot:
        state = State(indices, p=p, q=q)
        plt.close("all")
        state.plot()
        plt.savefig("{}/Desktop/state_break_0000".format(homedir), transparent=True)
        plt.close("all")

    break_state_count[i_trial, 0] = x0.copy()



    for i in range(1, n+1):
        #print("doing i: {}".format(i))
        cand_indices = indices[x0]

        trial_states, trial_ener = delta(x0, cand_indices)

        sel_mask = trial_ener == trial_ener.max()

        rand_idx = np.random.choice(sel_mask.sum())

        x0 = trial_states[sel_mask][rand_idx]

        break_state_count[i_trial,i] = x0.copy()
        if do_plot:
            state = State(indices[x0], p=p, q=q)
            plt.close("all")
            state.plot()
            plt.savefig("{}/Desktop/state_break_{:04d}".format(homedir, i), transparent=True)
            plt.close("all")



np.save('break_p_{:02d}_q_{:02d}_idx_{:03d}'.format(p, q, idx), break_state_count)

print ("Done with breaking...\n")
sys.stdout.flush()

## Build phobicity: Add methyls to pure polar #####
###################################################
print("Building hydrophobicity...\n")
sys.stdout.flush()

build_state_count = np.ones((n_trials, n+1, n), dtype=int)

for i_trial in range(n_trials):

    print("trial: {}".format(i_trial))

    x0 = np.zeros(n, dtype=bool)

    if do_plot:
        state = State(indices[x0], p=p, q=q)
        plt.close("all")
        state.plot()
        plt.savefig("{}/Desktop/state_build_0000".format(homedir), transparent=True)
        plt.close("all")

    build_state_count[i_trial, n] = x0

    for i in range(1, n+1):
        cand_indices = indices[~x0]

        trial_states, trial_ener = delta(x0, cand_indices)

        sel_mask = trial_ener == trial_ener.min()

        rand_idx = np.random.choice(sel_mask.sum())

        x0 = trial_states[sel_mask][rand_idx]
        build_state_count[i_trial, n-i] = x0.copy()

        if do_plot:
            state = State(indices[x0], p=p, q=q)

            plt.close("all")
            state.plot()
            plt.savefig("{}/Desktop/state_build_{:04d}".format(homedir, i), transparent=True)
            plt.close("all")

np.save('build_p_{:02d}_q_{:02d}_idx_{:03d}'.format(p, q, idx), build_state_count)

