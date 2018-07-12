# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm
from matplotlib import pyplot as plt

import numpy as np

import argparse

from IPython import embed

# Given bound simulation, find interfacial contact atoms by their dewetting in bound state 
#  (w.r.t unbound unbiased equil simulation)


# Find the normalized water density, exclude any buried atoms (by setting their normed density to 1.0),
#    and then locate dewet atoms that fall below dewetting thresh.

# Returns: tuple of : normed_rho: normalized water density for each atom (after excluding buried atoms)
#                     dewet_mask: mask identifiying dewetted atoms
def find_dewet_atoms(avg_water_ref, avg_water_targ, buried_mask, dewetting_thresh=0.5):

    normed_rho = avg_water_targ / avg_water_ref
    normed_rho[buried_mask] = 1.0


    return (normed_rho, (normed_rho < dewetting_thresh) )

# Find the number of TP, FP, TN, and FN from a list of predicted (boolean array) w.r.t. list of actual values (boolean array)
#
#  Returns tuple: (TP, FP, TN, FN)
def compare_pred_actual_acc(pred_vals, actual_vals):

    tp = pred_vals & actual_vals
    tn = ~pred_vals & ~actual_vals

    fp = pred_vals & ~actual_vals
    fn = ~pred_vals & actual_vals

    return (tp.sum(), fp.sum(), tn.sum(), fn.sum())

