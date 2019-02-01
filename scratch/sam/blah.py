from __future__ import division

import numpy as np
import MDAnalysis

from mdtools import MDSystem

import glob, os
from constants import k

beta = 1/(300*k)
thresh = 0.5

fnames = sorted(glob.glob('../phi_*/rho_data_dump_rad_6.0.dat.npz'))

phi_vals = [ fname.split('/')[1].split('_')[-1] for fname in fnames ]

sys = MDSystem('../phi_000/confout.gro', '../phi_000/confout.gro', sel_spec='resid 4435-4470 and (name S*)')
prot = sys.prot
hydropathy_mask = (prot.resnames == 'CH3')
k = hydropathy_mask.sum()
print('k={} ({:0.2f})'.format(k, k/36))

n_0 = np.load('../phi_000/rho_data_dump_rad_6.0.dat.npz')['rho_water'].mean(axis=0)


for fname, phi_val in zip(fnames, phi_vals):

    this_phi = float(phi_val)/10.0 #* beta
    print('phi: {:0.2f}'.format(this_phi))

    n_phi = np.load(fname)['rho_water'].mean(axis=0)

    rho_phi = n_phi/n_0

    dewet_mask = (rho_phi < thresh)
    n_dewet = dewet_mask.sum()
    dewet_phob = (dewet_mask & hydropathy_mask).sum()
    dewet_phil = (dewet_mask & ~hydropathy_mask).sum()

    print('  n_dewet: {}  frac_phob: {:0.2f}'.format(n_dewet, dewet_phob/n_dewet))

    prot.tempfactors = rho_phi
    prot.write('phi_{}_struct.pdb'.format(phi_val))