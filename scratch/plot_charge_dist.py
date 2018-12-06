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
charges = np.abs(rg.atoms.charges)

width = 0.01
bb = np.arange(0,1+width,width)
hist, bb = np.histogram(charges, bins=bb)

plt.bar(bb[:-1], hist, width=width, align='edge')
plt.show()

thresh = float(raw_input("choose charge threshold: "))

charge_dict = {}

for res in rg:
    print("Residue: {}".format(res.resname))
    chrge = 0
    charge_dict[res.resname] = {}
    #ag = res.atoms.select_atoms('not name H*')
    for atm in res.atoms:
        
        # Hydrophilic: -1;  Hydrophobic: 0
        hydrophil = -1 if np.abs(atm.charge) > thresh else 1
        
        atm.tempfactor = hydrophil
        charge_dict[res.resname][atm.name] = hydrophil
        print("  atm: {}  hv: {}  charge: {}".format(atm.name, hydrophil, atm.charge))
        chrge += atm.charge
        
    print("charge: {}".format(chrge))
    res.atoms.write("{}.pdb".format(res.resname))

with open('charge_assign.pkl', 'w') as f:
    pickle.dump(charge_dict, f)
