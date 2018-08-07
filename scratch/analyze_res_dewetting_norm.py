# After running dynamic_interface.py on a series of phi-ensemble simulations, 
#  i) find the buried atoms in the phi=0 simulation (anything that has n_water < thresh)
#  ii) Excluded buried atoms from every subsequent phi (by changing their tempfactors to thresh + 1)
#  iii) print out all atoms with tempfactors less than threshold for each phi

## **Assume we're in the directory of phi ensemble simuations**

import MDAnalysis
import os, glob
import numpy as np


thresh = 1
excl_thresh = 5

initial_file_paths = sorted(glob.glob('phi_*/dynamic_volume_norm.pdb'))

univ_wat0 = MDAnalysis.Universe('phi_000/dynamic_volume_water_avg.pdb') # avg number of water O atoms w/in 6 A of each protein atom
univ_prot0 = MDAnalysis.Universe('phi_000/dynamic_volume_solute_avg.pdb') # avg number of protein atoms w/in 6 A of each protein atom
buried_mask = univ_wat0.atoms.tempfactors < excl_thresh # exclude buried protein atoms - if they don't have many neighboring waters

for fpath in initial_file_paths:
    
    dirname = os.path.dirname(fpath)
    print("dir: {}".format(dirname))

    univ = MDAnalysis.Universe("{}/dynamic_volume_norm.pdb".format(dirname))

    univ_wat = MDAnalysis.Universe("{}/dynamic_volume_water_avg.pdb".format(dirname))
    univ_prot = MDAnalysis.Universe("{}/dynamic_volume_solute_avg.pdb".format(dirname))
    #Write out new file where buried atoms are masked

    univ.atoms.tempfactors = 0.5*( (univ_wat.atoms.tempfactors / univ_wat0.atoms.tempfactors) + (univ_prot.atoms.tempfactors / univ_prot0.atoms.tempfactors) )
    #univ.atoms.tempfactors = (univ_wat.atoms.tempfactors + univ_prot.atoms.tempfactors) / (univ_wat0.atoms.tempfactors + univ_prot0.atoms.tempfactors)
    univ.atoms[buried_mask].tempfactors = 1.0

    univ.atoms.write("{}/normed.pdb".format(dirname))

    # Get all atoms that are dry, not including buried ones
    indices = univ.atoms.tempfactors < thresh
    atms = univ.atoms[indices]
    print(" {} atoms with tempfactors less than {}".format(atms.n_atoms, thresh))
    sorted_idx = np.argsort(atms.tempfactors)

    # all the atoms, sorted by bfactor, with bfactor < thresh
    dry_atoms = atms[sorted_idx]
    for atm in dry_atoms:
        print("    ATM: {}  RES: {}  N: {}".format(atm.name, atm.residue, atm.bfactor))