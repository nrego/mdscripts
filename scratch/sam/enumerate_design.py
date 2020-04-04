from __future__ import division, print_function

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

from functools import reduce


log = logging.getLogger('mdtools.whamerr')

from IPython import embed




## Helper class for quickly finding the energies of all one-step moves from a given state
class GetDelta:
    def __init__(self, adj_mat, ext_count, alpha_n_cc, alpha_n_ce):
        self.adj_mat = adj_mat
        self.ext_count = ext_count
        self.alpha_n_cc = alpha_n_cc
        self.alpha_n_ce = alpha_n_ce


    def __call__(self, x0, cand_indices, prec=6):

        trial_energies = np.zeros_like(cand_indices).astype(float)
        trial_states = np.empty((cand_indices.size, x0.size), dtype=bool)
        x0_int = x0.astype(int)
        
        for i, cand_idx in enumerate(cand_indices):
            x1 = x0.copy()
            x1[cand_idx] = ~x1[cand_idx]

            x1_int = x1.astype(int)
            
            n_cc = 0.5 * np.linalg.multi_dot((x1_int, self.adj_mat, x1_int))
            n_ce = np.dot(x1_int, self.ext_count)

            trial_states[i] = x1
            trial_energies[i] = self.alpha_n_cc * n_cc + self.alpha_n_ce * n_ce
        

        return trial_states, np.round(trial_energies, prec)



class DesignDensity(ParallelTool):
    prog='sam pattern enumeration'
    description = '''\
Enumerate all patterns for greedy step-wise SAM pattern design.

Parallelized and dumps data when things get too large in mem (TODO)

Drops a checkpoint file at each round

-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    
    def __init__(self):
        super(DesignDensity,self).__init__()
        
        # Parallel processing by default (this is not actually necessary, but it is
        # informative!)
        self.wm_env.default_work_manager = self.wm_env.default_parallel_work_manager

        self.p = None
        self.q = None

        self.reg = None

        # If true, go pure philic->pure phobic
        self.build_phob = None

        self.delta = None

        # byte-keyed dictionaries of ints, pattern_as_byte=>count_of_pattern
        self.prev_state_count = None
        self.state_count = dict()

    # Total number of samples - sum of n_samples from each window
    @property
    def n(self):
        return self.p * self.q
    
    
    def add_args(self, parser):
        
        sgroup = parser.add_argument_group('Exhaustively enumerate sam design patterns')
        sgroup.add_argument('--reg', default='sam_reg_coef.npy', type=str,
                            help='Input file with regression coefs (default: %(default)s)')
        sgroup.add_argument('-p', default=6, type=int,
                            help='p (default: %(default)s)')
        sgroup.add_argument('-q', default=6, type=int,
                            help='q (default: %(default)s)')
        sgroup.add_argument('--build-phob', action='store_true', 
                            help='If true, go philic => phobic')
        sgroup.add_argument('--cpi', type=str, help='checkpoint telling us which round we are starting with')


    def process_args(self, args):

        self.p = args.p
        self.q = args.q
        tmp_state = State(np.array([], dtype=int), self.p, self.q)

        self.reg = np.load('sam_reg_coef.npy').item()

        alpha1, alpha2, alpha3 = self.reg.coef_

        alpha_n_cc = alpha2
        alpha_n_ce = alpha2 - alpha3

        self.delta = GetDelta(tmp_state.adj_mat, tmp_state.ext_count, alpha_n_cc, alpha_n_ce)

        self.build_phob = args.build_phob

        if args.cpi:
            with open(args.cpi, 'rb') as f:
                self.prev_state_count = pickle.load(f)
        else:
            self.prev_state_count = dict()

            if self.build_phob:
                x0 = np.zeros(self.n, dtype=bool).tobytes()
            else:
                x0 = np.ones(self.n, dtype=bool).tobytes()

            self.prev_state_count[x0] = 1

    def update_state_count(self, states, mult):
        for newx in states:
            try:
                self.state_count[newx.tobytes()] += mult
            except KeyError:
                self.state_count[newx.tobytes()] = mult

    def go(self):
        indices = np.arange(self.n, dtype=int)

        prev_states = np.array([np.fromstring(bstr, dtype=bool) for bstr in list(self.prev_state_count.keys())], dtype=bool)
        prev_counts = np.array(list(self.prev_state_count.values()))

        # Number of methyls in each prev config
        prev_num_methyl = prev_states[0].sum()
        if self.build_phob:
            this_num_methyl = prev_num_methyl + 1
        else:
            this_num_methyl = prev_num_methyl - 1

        mode = 'build phob' if self.build_phob else 'break phob'
        log.info("Starting!\n")
        log.info("this round: {} to {} methyls ({})".format(prev_num_methyl, this_num_methyl, mode))
        log.info("we have {} states to process".format(prev_states.shape[0]))


        ## TODO: Make parallel!
        n_workers = self.work_manager.n_workers or 1
        batch_size = prev_states.shape[0] // n_workers

        for i, x0 in enumerate(prev_states):

            if i % 1000 == 0:
                log.info("processing state {}".format(i+1))
            if self.build_phob:
                cand_indices = indices[~x0]
            else:
                cand_indices = indices[x0]

            mult = self.prev_state_count[x0.tobytes()]

            trial_states, trial_ener = self.delta(x0, cand_indices)


            # Select states that cause biggest gain (break_phob) or loss (build_phob) in FE
            if self.build_phob:
                trial_ener = -trial_ener

            sel_mask = trial_ener == trial_ener.max()

            ## For each of the (possibly multiple) new states, 
            self.update_state_count(trial_states[sel_mask], mult)

        # We're done enumerating where the prev states lead. now output
        print("finished !")

        with open('kc_{:04d}.pkl'.format(this_num_methyl), 'wb') as fout:
            pickle.dump(self.state_count, fout)


if __name__=='__main__':
    DesignDensity().main()
