import numpy as np
import os
import matplotlib as mpl
import MDAnalysis
import pickle


## Classify each surface atom as polar, non-polar, locally convex/concave
#    Compare these classifications to their beta phi i star's

univ = MDAnalysis.Universe('just_prot/topol.tpr', 'just_prot/out.gro')

buried_mask = np.loadtxt("beta_phi_000/buried_mask.dat", dtype=bool)
surf_mask = ~buried_mask
atoms = univ.select_atoms('protein and not name H*')[surf_mask]
beta_phi_i_star = np.loadtxt("beta_phi_star_prot.dat")[surf_mask]
curv = np.loadtxt("atomic_curvature.dat")[surf_mask]
hydropathy = np.loadtxt("hydropathy_mask.dat")[surf_mask]
contact_mask = np.loadtxt("actual_contact_mask.dat", dtype=bool)[surf_mask] 

non_polar_mask = hydropathy.astype(bool)


with open('charge_dict.pkl', 'rb') as fin:
    charge_dict = pickle.load(fin)

atm_charges = np.round(atoms.charges, 4)


#plt.plot(curv[non_polar_mask], beta_phi_i_star[non_polar_mask], 'x')
#plt.show()

#plt.plot(curv[~non_polar_mask], beta_phi_i_star[~non_polar_mask], '^')
#plt.show()