# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm

base_univ = MDAnalysis.Universe('phi_000/confout.gro')
base_prot_group = base_univ.select_atoms('protein and not name H*')
univ = MDAnalysis.Universe('phi_000/dynamic_volume_water_avg.pdb')
prot_atoms = univ.atoms
assert base_prot_group.n_atoms == prot_atoms.n_atoms

# Note: 0-indexed!
all_atom_lookup = base_prot_group.indices # Lookup table from prot_atoms index to all-H index
atm_indices = prot_atoms.indices

max_water = 33.0 * (4./3.)*np.pi*(0.6)**3

# To determine if atom buried (if avg waters fewer than thresh) or not
avg_water_thresh = 7
# atom is 'dewet' if its average number of is less than this percent of
#   its value at phi=0
per_dewetting_thresh = 0.6

ds_0 = np.load('phi_000/rho_data_dump.dat.npz')

# shape: (n_atoms, n_frames)
rho_solute0 = ds_0['rho_solute'].T
rho_water0 = ds_0['rho_water'].T

assert rho_solute0.shape[0] == rho_water0.shape[0] == prot_atoms.n_atoms
n_frames = rho_water0.shape[1]

# averages per heavy
rho_avg_solute0 = rho_solute0.mean(axis=1)
rho_avg_water0 = rho_water0.mean(axis=1)

# Find surface atoms...
surf_mask = rho_avg_water0 > avg_water_thresh
np.savetxt('surf_mask.dat', surf_mask, fmt='%d')
rho_solute0 = rho_solute0[surf_mask]
rho_water0 = rho_water0[surf_mask]
rho_avg_solute0 = rho_avg_solute0[surf_mask]
rho_avg_water0 = rho_avg_water0[surf_mask]

surf_indices = atm_indices[surf_mask]

# variance and covariance
delta_rho_water0 = rho_water0 - rho_avg_water0[:,np.newaxis]
cov_rho_water0 = np.dot(delta_rho_water0, delta_rho_water0.T) / n_frames
rho_var_water0 = cov_rho_water0.diagonal()

del ds_0

paths = sorted(glob.glob('phi*/rho_data_dump.dat.npz'))

for fpath in paths:
    dirname = os.path.dirname(fpath)

    all_h_univ = MDAnalysis.Universe('{}/confout.gro'.format(dirname))
    all_h_univ.atoms.bfactors = 0
    all_prot_atoms = all_h_univ.select_atoms('protein and not name H*')
    univ = MDAnalysis.Universe('{}/dynamic_volume_water_avg.pdb'.format(dirname))

    # mask the buried atoms
    univ.atoms[~surf_mask].bfactors = max_water
    all_prot_atoms[~surf_mask].bfactors = max_water
    univ.atoms.write('{}/surf_masked.pdb'.format(dirname))

    ds = np.load(fpath)
    rho_solute = (ds['rho_solute'].T)[surf_mask]
    rho_water = (ds['rho_water'].T)[surf_mask]
    del ds

    assert rho_water.shape[0] == surf_mask.sum()
    # can't assume all datasets have the same number of frames in analysis
    #    (though they should!)
    n_frames = rho_water.shape[1]

    rho_avg_solute = rho_solute.mean(axis=1)
    rho_avg_water = rho_water.mean(axis=1)

    # percentage each atom is dewet (w.r.t. its phi=0.0 value)
    per_dewet = (rho_avg_water/rho_avg_water0)
    univ.atoms[surf_mask].bfactors = per_dewet
    all_prot_atoms[surf_mask].bfactors = per_dewet
    univ.atoms.write('{}/per_dewet.pdb'.format(dirname))
    all_h_univ.atoms.write('{}/all_per_dewet.pdb'.format(dirname))

    dewet_mask = per_dewet < per_dewetting_thresh
    univ.atoms.bfactors = 1
    all_prot_atoms.bfactors = 1
    univ.atoms[surf_mask][dewet_mask].bfactors = 0
    all_prot_atoms[surf_mask][dewet_mask].bfactors = 0
    univ.atoms.write('{}/dewet_atoms.pdb'.format(dirname))
    all_h_univ.atoms.write('{}/all_dewet_atoms.pdb'.format(dirname))

    print("dir: {}  n_dewet: {} of {}".format(dirname, dewet_mask.sum(), surf_mask.sum()))






