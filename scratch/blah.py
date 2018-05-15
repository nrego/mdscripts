# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm

dat = np.loadtxt('logweights.dat')
fnames = sorted(glob.glob('*/phiout.dat'))



bc = dat[:,0]
pdist = np.exp(-dat[:,1])
pdist /= np.trapz(pdist, bc)

neglogpdist = -np.log(pdist)
mask = ~np.ma.masked_invalid(neglogpdist).mask

beta = 1/(k*300.)

phi_vals = np.arange(0,10.1,0.1)

avg_N = []
var_N = []

for phi in phi_vals:
    bias = beta*phi*bc
    bias_fe = neglogpdist[mask] + bias[mask]
    bias_fe -= bias_fe.min()

    bias_pdist = np.exp(-bias_fe)

    bias_pdist /= np.trapz(bias_pdist, bc[mask])

    this_avg_N = np.trapz(bias_pdist*bc[mask], bc[mask])
    this_avg_N_sq = np.trapz(bias_pdist*bc[mask]**2, bc[mask])

    avg_N.append(this_avg_N)
    var_N.append(this_avg_N_sq - this_avg_N**2)
