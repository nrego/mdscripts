univ = MDAnalysis.Universe('pred_contact.pdb')
prot = univ.atoms

buried_mask = np.loadtxt('buried_mask.dat', dtype=bool)
pred_mask = np.loadtxt('pred_contact_mask.dat', dtype=bool)
h_mask = np.loadtxt('../../bound/hydropathy_mask.dat', dtype=bool)

prot.tempfactors = -1
prot[h_mask].tempfactors = 1
prot[~pred_mask].tempfactors = -2

prot.write('pred_h.pdb')