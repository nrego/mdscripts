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

from mdtools import MDSystem


## Run from phi_sims directory

# Plots out the distribution of <n_i>_0, prompts user for cutoff threshold, excluded buried atoms.
# Excludes buried atoms, writes out buried and surface masks, and also does shit with structure file to show buried atoms.
infile = 'unbound/uim/rho_data_dump_rad_6.0.dat.npz'
univ_infile = 'unbound/uim/dynamic_volume_water_avg.pdb'
binspace = 1
# Buried if <= this 
avg_excl = 5


