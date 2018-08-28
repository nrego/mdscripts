from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed

import cPickle as pickle

univ = MDAnalysis.Universe('top.tpr', 'conf.gro')
univ.add_TopologyAttr('tempfactor')

rg = univ.residues[1:-1]
ag = rg.atoms

charge_dict = {}

for res in rg:
    print("Residue: {}".format(res.resname))
    chrge = 0
    charge_dict[res.resname] = {}
    ag = res.atoms.select_atoms('not name H*')
    for atm in ag:
        #embed()
        this_charge = atm.charge
        this_dipole = np.array([0,0,0.0])

        # loop thru hydrogen atoms
        for bond_h in atm.bonded_atoms:
            if bond_h.name[0] == 'H' or (atm.name=='C' and bond_h.name =='O') or (atm.name=='O' and bond_h.name =='C'):
        
                bond_vec = bond_h.position - atm.position
                q = bond_h.charge
                this_charge += q

                this_dipole += q * bond_vec


        hydrophillic = (np.linalg.norm(this_dipole) > 0.25) or (np.abs(this_charge) > 0.25)
        hydrophil = 1 if hydrophillic else -1
        atm.tempfactor = hydrophil
        for bond_h in atm.bonded_atoms:
            if bond_h.name[0] == 'H':
                bond_h.tempfactor = hydrophil
        #charge_dict[res.resname][atm.name] = hydrophil
        print("  atm: {}  mag_dipole: {}  charge: {}".format(atm.name, np.linalg.norm(this_dipole), this_charge))
    

    res.atoms.write("{}.pdb".format(res.resname))

#with open('charge_assign.pkl', 'w') as f:
#    pickle.dump(charge_dict, f)
