# After running dynamic_interface.py on a series of phi-ensemble simulations, 
#  i) find the buried atoms in the phi=0 simulation (anything that has n_water < thresh)
#  ii) Excluded buried atoms from every subsequent phi (by changing their bfactors to thresh + 1)
#  iii) print out all atoms with bfactors less than threshold for each phi

## **Assume we're in the directory of phi ensemble simuations**

import MDAnalysis
import os, glob
import numpy as np


thresh = 0.6
excl_thresh = 1.0

initial_file_paths = sorted(glob.glob('phi_*/dynamic_volume_norm.pdb'))

univ = MDAnalysis.Universe('phi_000/dynamic_volume_water_avg.pdb')
buried_mask = univ.atoms.bfactors < excl_thresh

for fpath in initial_file_paths:
    
    dirname = os.path.dirname(fpath)
    print("dir: {}".format(dirname))
    univ = MDAnalysis.Universe(fpath)
    univ.atoms[buried_mask].bfactors = 1.0
    #Write out new file where buried atoms are masked

    # Get all atoms that are dry, not including buried ones
    indices = univ.atoms.bfactors < thresh
    atms = univ.atoms[indices]
    print(" {} atoms with bfactors less than {}".format(atms.n_atoms, thresh))
    sorted_idx = np.argsort(atms.bfactors)

    # all the atoms, sorted by bfactor, with bfactor < thresh
    dry_atoms = atms[sorted_idx]
    for atm in dry_atoms:
        print("    ATM: {}  RES: {}  N: {}".format(atm.name, atm.residue, atm.bfactor))