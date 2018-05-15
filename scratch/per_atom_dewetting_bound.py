# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm

# Determine native contacts by dewetted atoms in the bound
#   state

sel_spec = "segid seg_0_Protein_chain_A and not name H*"
#sel_spec = "protein and not name H*"

base_univ = MDAnalysis.Universe('phi_000/confout.gro')
base_prot_group = base_univ.select_atoms('protein and not name H*')
univ = MDAnalysis.Universe('phi_000/dynamic_volume_water_avg.pdb')
prot_atoms = univ.atoms
assert base_prot_group.n_atoms == prot_atoms.n_atoms

try:
    nat_contacts = MDAnalysis.Universe('nat_contacts.pdb')
    
except:
    nat_contacts = None
# weight for calculating per-atom norm density
wt = 0.5 

# Note: 0-indexed!
all_atom_lookup = base_prot_group.indices # Lookup table from prot_atoms index to all-H index
atm_indices = prot_atoms.indices

max_water = 33.0 * (4./3.)*np.pi*(0.6)**3

# To determine if atom buried (if avg waters fewer than thresh) or not
avg_water_thresh = 6
# atom is 'dewet' if its average number of is less than this percent of
#   its value at phi=0
per_dewetting_thresh = 0.5

ds_0 = np.load('phi_000/rho_data_dump.dat.npz')

# shape: (n_atoms, n_frames)
rho_solute0 = ds_0['rho_solute'].T
rho_water0 = ds_0['rho_water'].T

assert rho_solute0.shape[0] == rho_water0.shape[0] == prot_atoms.n_atoms
n_frames = rho_water0.shape[1]

# averages per heavy
rho_avg_solute0 = rho_solute0.mean(axis=1)
rho_avg_water0 = rho_water0.mean(axis=1)

max_solute = rho_avg_solute0.max()
max_water = rho_avg_water0.max()

buried_mask = rho_avg_water0 < avg_water_thresh
surf_mask = ~buried_mask

# variance and covariance
delta_rho_water0 = rho_water0 - rho_avg_water0[:,np.newaxis]
cov_rho_water0 = np.dot(delta_rho_water0, delta_rho_water0.T) / n_frames
rho_var_water0 = cov_rho_water0.diagonal()

del ds_0

## Bound complex
fpath = './rho_data_dump.dat.npz'
dirname = os.path.dirname(fpath)

all_h_univ = MDAnalysis.Universe('prod.tpr', 'em.gro')
all_h_univ.atoms.bfactors = 0
all_prot_atoms = all_h_univ.select_atoms(sel_spec)
univ = MDAnalysis.Universe('{}/dynamic_volume_water_avg.pdb'.format(dirname))

ds = np.load(fpath)
rho_solute = (ds['rho_solute'].T)
rho_water = (ds['rho_water'].T)
del ds

# can't assume all datasets have the same number of frames in analysis
#    (though they should!)
n_frames = rho_water.shape[1]

rho_avg_solute = rho_solute.mean(axis=1)
rho_avg_water = rho_water.mean(axis=1)

# New, normalized water and protein density
#   hopefully accounts for case when exposed atom becomes buried at phi>0
#   (and would possibly be a false dewet positive otherwise)
per_dewet = wt*(rho_avg_water/rho_avg_water0) + (1-wt)*(rho_avg_solute/rho_avg_solute0)
#per_dewet = (rho_avg_water/rho_avg_water0)
rho_norm = (rho_avg_water/max_water) + (rho_avg_solute/max_solute)

univ.atoms.bfactors = rho_norm
univ.atoms[buried_mask].bfactors = 1
all_prot_atoms.bfactors = rho_norm
all_prot_atoms[buried_mask].bfactors = 1

univ.atoms.write('{}/global_norm.pdb'.format(dirname))
all_h_univ.atoms.write('{}/all_global_norm.pdb'.format(dirname))

univ.atoms.bfactors = per_dewet
univ.atoms[buried_mask].bfactors = 1
all_prot_atoms.bfactors = per_dewet
all_prot_atoms[buried_mask].bfactors = 1

univ.atoms.write('{}/per_norm.pdb'.format(dirname))
all_h_univ.atoms.write('{}/all_per_norm.pdb'.format(dirname))

dewet_mask = (per_dewet < per_dewetting_thresh) & surf_mask
univ.atoms.bfactors = 1
univ.atoms[dewet_mask].bfactors = 0
all_prot_atoms.bfactors = 1
all_prot_atoms[dewet_mask].bfactors = 0
univ.atoms.write('{}/per_atom_dewet.pdb'.format(dirname))
all_h_univ.atoms.write('{}/all_per_atom_dewet.pdb'.format(dirname))
#print("dir: {}  n_dewet (per-atom): {} of {}".format(dirname, dewet_mask.sum(), prot_atoms.n_atoms))

dewet_mask = (rho_norm < per_dewetting_thresh) & surf_mask
univ.atoms.bfactors = 1
univ.atoms[dewet_mask].bfactors = 0
all_prot_atoms.bfactors = 1
all_prot_atoms[dewet_mask].bfactors = 0
univ.atoms.write('{}/norm_dewet.pdb'.format(dirname))
all_h_univ.atoms.write('{}/all_norm_dewet.pdb'.format(dirname))
print("dir: {}  n_dewet (global norm): {} of {}".format(dirname, dewet_mask.sum(), prot_atoms.n_atoms))



## Check dewetted atom contacts versus distance contacts 
if nat_contacts is not None:
    thresholds = np.arange(0, 1.05, 0.05)

    contact_mask = nat_contacts.atoms.bfactors == 0
    print("total native contacts: {}".format(contact_mask.sum()))

    roc = np.zeros((thresholds.size, 2))

    for i, thresh in enumerate(thresholds):
        print("S={}".format(thresh))
        dewet_mask = rho_norm < thresh

        true_pos = dewet_mask & contact_mask
        false_pos = dewet_mask & ~contact_mask

        true_neg = ~dewet_mask & ~contact_mask
        false_neg = ~dewet_mask & contact_mask

        assert dewet_mask.sum() == true_pos.sum() + false_pos.sum()

        print("  total predicted: {}  true pos: {}  false pos: {}  true neg: {}  false_neg: {}".format(dewet_mask.sum(), true_pos.sum(), false_pos.sum(), true_neg.sum(), false_neg.sum()))

        tpr = true_pos.sum() / ( true_pos.sum()+false_neg.sum() )
        fpr = false_pos.sum() / ( true_neg.sum()+false_pos.sum() )
        acc = (true_pos.sum() + true_neg.sum())/univ.atoms.n_atoms
        f1 = (2*true_pos.sum())/(2*true_pos.sum()+false_pos.sum()+false_neg.sum())
        print("  TPR: {}  FPR: {}".format(tpr,fpr))
        print("  ACC: {}".format(acc))
        print("  F1: {}".format(f1))
        roc[i] = fpr, tpr


    plt.plot(roc[:,0], roc[:,1], '-o')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.title(r'$\phi={}$'.format(phi_val))
    plt.tight_layout()

    roc_arr = np.hstack((thresholds[:,None], roc)).squeeze()
    np.savetxt('{}/roc.dat'.format(dirname), roc_arr)
