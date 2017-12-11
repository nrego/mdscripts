# After running dynamic_interface.py on a series of phi-ensemble simulations, 
#  i) find the buried atoms in the phi=0 simulation (anything that has n_water < thresh)
#  ii) Excluded buried atoms from every subsequent phi (by changing their bfactors to thresh + 1)
#  iii) print out all atoms with bfactors less than threshold for each phi

## **Assume we're in the directory of phi ensemble simuations**

import MDAnalysis
import os, glob
import numpy as np

thresh = 1

initial_file_paths = sorted(glob.glob('phi_*/dynamic_volume_targ_water_avg.pdb'))

univ = MDAnalysis.Universe(initial_file_paths[0])
buried_mask = univ.atoms.bfactors < thresh
initial_waters = univ.atoms.bfactors

for fpath in initial_file_paths:
    
    dirname = os.path.dirname(fpath)
    print("dir: {}".format(dirname))
    univ = MDAnalysis.Universe(fpath)
    univ.atoms[buried_mask].bfactors = thresh+1.0
    # change in waters
    #univ.atoms.bfactors = initial_waters - univ.atoms.bfactors 

    #Write out new file where buried atoms are masked
    univ.atoms.write("{}/waters.pdb".format(dirname))

    # Get all atoms that are dry, not including buried ones
    indices = univ.atoms.bfactors < thresh
    atms = univ.atoms[indices]
    print(" {} atoms dewetted".format(atms.n_atoms))
    sorted_idx = np.argsort(atms.bfactors)

    # all the atoms, sorted by bfactor, with bfactor < thresh
    dry_atoms = atms[sorted_idx][::-10]
    for atm in dry_atoms:
        print("    ATM: {}  RES: {}  N: {}".format(atm.name, atm.residue, atm.bfactor))