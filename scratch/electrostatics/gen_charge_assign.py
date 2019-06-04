from __future__ import division, print_function
import numpy as np
import MDAnalysis 
import cPickle as pickle


univ = MDAnalysis.Universe('top.tpr', 'conf.gro')
univ.add_TopologyAttr('tempfactors')
# Exclude termini
atoms = univ.residues[1:-1].atoms
assert atoms.residues.n_residues == 20

charges = atoms.charges
bw = 0.01
bb = np.arange(0, 1+bw, bw)
hist, _ = np.histogram(np.abs(charges), bb, normed=True)
fig, ax = plt.subplots()
ax.bar(bb[:-1], hist, width=bw, align='edge')
ax.set_xticks(np.arange(0,1.1,0.1))
fig.show()

charge_dict = {}

for res in atoms.residues:
    charge_dict[res.resname] = {}
    for atm in res.atoms:
        abs_charge = np.abs(atm.charge)
        philicity = -1 if abs_charge > 0.25 else 0
        charge_dict[res.resname][atm.name] = philicity
        atm.tempfactor = philicity

    res.atoms.write('{}.pdb'.format(res.resname))

with open('charge_assign.pkl', 'w') as fout:
    pickle.dump(charge_dict, fout)