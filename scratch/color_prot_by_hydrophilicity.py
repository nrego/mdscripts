from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed

import cPickle as pickle

with open('charge_assign.pkl', 'r') as f:
    charge_assign = pickle.load(f)

rho_dat = np.load('rho_data_dump_rad_6.0.dat.npz')['rho_water'].mean(axis=0)

buried_mask = rho_dat < 5
n_buried = buried_mask.sum()
surf_mask = ~buried_mask
n_surf = surf_mask.sum()

univ = MDAnalysis.Universe('confout.gro')
univ.add_TopologyAttr('tempfactor')

prot = univ.select_atoms('protein')
prot_heavies = univ.select_atoms('protein and not name H*')
n_atoms = prot.n_atoms
print("{} protein atoms".format(n_atoms))
prot.tempfactors = 0
prot_heavies[buried_mask].tempfactors = 1
prot_heavies.write('buried.pdb')
prot.tempfactors = 0
#assert prot.n_atoms == rho_dat.size

for atm in prot:
    try:
        hydrophil = charge_assign[atm.resname][atm.name]
    except:
        hydrophil = float(raw_input('enter hydrophilicity for atom {} of {} (-1 for hydrophobic, 1 for hydrophilic):  '.format(atm.name, atm.resname)))
        charge_assign[atm.resname][atm.name] = hydrophil
    atm.tempfactor = hydrophil

prot.write('prot_by_charge.pdb')

hydrophil_mask_all = (prot.atoms.tempfactors == 1)
hydrophob_mask_all = (prot.atoms.tempfactors == -1)

n_hydrophil_all = hydrophil_mask_all.sum()
n_hydrophob_all = hydrophob_mask_all.sum()

print("Total atoms: {}".format(n_atoms))
print("  N hydrophilic: {}".format(n_hydrophil_all))
print("  N hydrophobic: {}".format(n_hydrophob_all))
print("  frac hydrophilic: {}".format(n_hydrophil_all / n_atoms))

# Just heavy atoms
hydrophil_mask = (prot_heavies.tempfactors == 1)
hydrophob_mask = (prot_heavies.tempfactors == -1)

n_hydrophil = hydrophil_mask.sum()
n_hydrophob = hydrophob_mask.sum()

n_surf_hydrophil = (surf_mask & hydrophil_mask).sum()
n_surf_hydrophob = (surf_mask & hydrophob_mask).sum()

n_buried_hydrophil = (buried_mask & hydrophil_mask).sum()
n_buried_hydrophob = (buried_mask & hydrophob_mask).sum()

assert n_hydrophob+n_hydrophil == prot_heavies.n_atoms

print("Heavy atoms: {}".format(prot_heavies.n_atoms))
print("  N hydrophilic: {}".format(n_hydrophil))
print("  N hydrophobic: {}".format(n_hydrophob))
print("  frac hydrophilic: {}".format(n_hydrophil / prot_heavies.n_atoms))

print("  N surf: {}".format(n_surf))
print("      hydrophil: {}".format(n_surf_hydrophil))
print("      hydrophob: {}".format(n_surf_hydrophob))
print("      frac hydrophilic: {}".format(n_surf_hydrophil/n_surf))
print("  N buried: {}".format(n_buried))
print("      hydrophil: {}".format(n_buried_hydrophil))
print("      hydrophob: {}".format(n_buried_hydrophob))
print("      frac hydrophilic: {}".format(n_buried_hydrophil/n_buried))

prot_heavies[buried_mask].tempfactors = -2 
prot_heavies.write('prot_heavies_by_charge.pdb')

with open('charge_assign.pkl', 'w') as f:
    pickle.dump(charge_assign, f)

header='n_tot   n_tot_hydrophil   n_tot_hydrophob   n_heavy   n_heavy_surf  n_heavy_surf_hydrophil  n_heavy_surf_hydrophob   n_heavy_buried  n_heavy_buried_hydrophil  n_heavy_buried_hydrophob'
dat_arr = np.array([n_atoms, n_hydrophil_all, n_hydrophob_all, prot_heavies.n_atoms, n_surf, n_surf_hydrophil, n_surf_hydrophob, n_buried, n_buried_hydrophil, n_buried_hydrophob])

np.savetxt('struct_data.dat', dat_arr, header=header)




