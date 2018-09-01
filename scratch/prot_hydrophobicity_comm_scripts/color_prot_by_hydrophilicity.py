from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed

from mdtools import MDSystem

import cPickle as pickle

with open('charge_assign.pkl', 'r') as f:
    charge_assign = pickle.load(f)

rho_dat = np.load('rho_data_dump_rad_6.0.dat.npz')['rho_water'].mean(axis=0)

sys = MDSystem('top.tpr', 'cent.gro', rho_dat)
sys.find_buried(nb=5)
sys.assign_hydropathy(charge_assign)

sys.prot.write('prot_by_charge.pdb')
sys.prot_h.write('prot_heavies_by_charge.pdb')

print("Total atoms: {}".format(sys.n_prot_tot))
print("N surf: {}".format(sys.n_surf))
print("  N hydrophilic: {}".format(sys.n_phil))
print("  N hydrophobic: {}".format(sys.n_phob))
print("  frac hydrophilic: {}".format(sys.n_phil/sys.n_surf))

print("Heavy atoms: {}".format(sys.n_prot_h_tot))
print("N surf: {}".format(sys.n_surf_h))
print("  N hydrophilic: {}".format(sys.n_phil_h))
print("  N hydrophobic: {}".format(sys.n_phob_h))
print("  frac hydrophilic: {}".format(sys.n_phil_h/sys.n_surf_h))

