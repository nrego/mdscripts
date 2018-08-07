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


def find_buried(infile, univ_infile, avg_excl=5):
    univ = MDAnalysis.Universe(univ_infile)

    water_avg_0 = np.load(infile)['rho_water'].mean(axis=0)

    print("n prot atoms: {}".format(water_avg_0.size))

    bb = np.arange(0,50,binspace)
    hist, bb = np.histogram(water_avg_0, bins=bb, normed=True)
    cum_hist = np.diff(bb)[0]*np.cumsum(hist)

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.bar(bb[:-1], hist, align='edge', width=binspace)
    ax2.plot(bb[1:], cum_hist, '-k')

    #plt.show()


    print("excluding all with atoms with <= {} average waters".format(avg_excl))

    buried_mask = water_avg_0 <= avg_excl
    surf_mask = ~buried_mask

    print("{} out of {} atoms excluded ({} surface remain)".format(buried_mask.sum(), water_avg_0.size, surf_mask.sum()))

    univ.atoms[buried_mask].tempfactors = 50

    univ.atoms.write('buried_masked_equil.pdb')
    np.savetxt('buried_mask.dat', buried_mask, fmt='%1d')
    np.savetxt('surf_mask.dat', surf_mask, fmt='%1d')




## Run from phi_sims directory

# Plots out the distribution of <n_i>_0, prompts user for cutoff threshold, excluded buried atoms.
# Excludes buried atoms, writes out buried and surface masks, and also does shit with structure file to show buried atoms.
infile = 'unbound/uim/rho_data_dump_rad_6.0.dat.npz'
univ_infile = 'unbound/uim/dynamic_volume_water_avg.pdb'
binspace = 1
# Buried if <= this 
avg_excl = 5

find_buried(infile, univ_infile, avg_excl)