# After running dynamic_interface.py on a series of phi-ensemble simulations, 
#  i) find the buried atoms in the phi=0 simulation (anything that has n_water < thresh)
#  ii) Excluded buried atoms from every subsequent phi (by changing their bfactors to thresh + 1)
#  iii) print out all atoms with bfactors less than threshold for each phi

## **Assume we're in the directory of phi ensemble simuations**

import MDAnalysis
import os, glob
import numpy as np
import cPickle as pickle

thresh = 0.75

with open('../pdb/contacts.pkl', 'r') as f:
    all_contacts = pickle.load(f)

with open('../pdb/hyd_contacts.pkl', 'r') as f:
    hyd_contacts = pickle.load(f)

hyd_keys = np.array(hyd_contacts.keys())
all_keys = np.array(all_contacts.keys())
#initial_file_paths = sorted(glob.glob('phi_*/dynamic_volume_water_avg.pdb'))
initial_file_paths = sorted(glob.glob('phi_*/dynamic_volume_norm.pdb'))

univ = MDAnalysis.Universe(initial_file_paths[0])
buried_mask = univ.atoms.bfactors < thresh
initial_waters = univ.atoms.bfactors

all_indices = np.arange(univ.atoms.n_atoms)
for fpath in initial_file_paths:
    
    dirname = os.path.dirname(fpath)
    print("dir: {}".format(dirname))
    univ = MDAnalysis.Universe(fpath)
    #univ.atoms[buried_mask].bfactors = thresh+1.0
    # change in waters
    #univ.atoms.bfactors = initial_waters - univ.atoms.bfactors 

    #Write out new file where buried atoms are masked
    #univ.atoms.write("{}/waters.pdb".format(dirname))

    # Get all atoms that are dry, not including buried ones
    indices = univ.atoms.bfactors < thresh
    atms = univ.atoms[indices]
    print(" {} atoms dewetted".format(atms.n_atoms))
    sorted_idx = np.argsort(atms.bfactors)

    # all the atoms, sorted by bfactor, with bfactor < thresh
    dry_atoms = atms[sorted_idx][:10]
    for atm in dry_atoms:
        print("    ATM: {}  RES: {}  N: {}".format(atm.name, atm.residue, atm.bfactor))

    cand_indices = all_indices[indices]

    in_all_contacts = np.zeros_like(cand_indices, dtype=bool)
    in_hyd_contacts = np.zeros_like(cand_indices, dtype=bool)

    for i, idx in enumerate(cand_indices):
        if idx in all_keys:
            in_all_contacts[i] = True
            if idx in hyd_keys:
                in_hyd_contacts[i] = True

    n_true_positives = in_hyd_contacts.sum()
    n_false_positives = cand_indices.size - n_true_positives
    n_false_negatives = hyd_keys.size - n_true_positives

    print("n_true_positives: {}".format(n_true_positives))
    #print("n_false_negatives: {}".format(n_false_negatives))
    print("n_false_positives: {}".format(n_false_positives))
