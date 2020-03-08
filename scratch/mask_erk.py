from scratch.prot_pred import align

# From hotspot/2erk/old_prot_pred/bound directory
univ = MDAnalysis.Universe('actual_contact.pdb')
prot = univ.atoms
prot.tempfactors = 0

# Human ERK bound to PEA-15 DED
u2 = MDAnalysis.Universe('../pea/dimer/human_erk.pdb')
p2 = u2.residues.atoms

# Align atoms to human ERK2
main_mask, other_mask = align(prot, p2)

assert prot[main_mask].n_atoms == p2[other_mask].n_atoms
assert np.array_equal(prot[main_mask].names, p2[other_mask].names)

# Now find pea-15 binding to human erk2, map onto rat erk2
contact = np.load('../pea/dimer/min_dist_neighbor.dat.npz')['min_dist'].mean(axis=0) < 5
mapped_contact_mask = contact[other_mask]

buried_mask = np.loadtxt('../pred_reweight/beta_phi_000/buried_mask.dat', dtype=bool)

# Phosphorylated thr and tyr
prot.tempfactors = 0
ag = prot.select_atoms('resid 183 or resid 185')
ag.tempfactors = 1
mask_phospho = prot.tempfactors.astype(bool)
np.savetxt('phospho_mask.dat', mask_phospho, fmt='%1d')

# Catalytic residues
prot.tempfactors = 0
ag = prot.select_atoms('resid 145 or resid 146 or resid 147 or resid 149')
ag.tempfactors = 1
mask_cat = prot.tempfactors.astype(bool)
np.savetxt('cat_mask.dat', mask_cat, fmt='%1d')

# DRS
prot.tempfactors = 0
ag = prot.select_atoms('resid 113 or resid 119 or resid 123 or resid 126 or resid 155 or resid 157 or resid 158 or resid 316 or resid 319')
ag.tempfactors = 1
mask_drs = prot.tempfactors.astype(bool)
np.savetxt('drs_mask.dat', mask_drs, fmt='%1d')

# FRS
prot.tempfactors = 0
ag = prot.select_atoms('resid 197 or resid 198 or resid 231 or resid 232 or resid 235 or resid 261')
ag.tempfactors = 1
mask_frs = prot.tempfactors.astype(bool)
np.savetxt('frs_mask.dat', mask_frs, fmt='%1d')


# PEA-15 binding
prot.tempfactors = 0
ag = prot[main_mask][mapped_contact_mask]
ag.tempfactors = 1
mask_pea_bind = prot.tempfactors.astype(bool)
np.savetxt('actual_contact_mask.dat', mask_pea_bind, fmt='%1d')

prot.tempfactors = 0
prot[mask_phospho].tempfactors = 1
prot[mask_cat].tempfactors = 2
prot[mask_drs].tempfactors = 3
prot[mask_frs].tempfactors = 4
#prot[mask_pea_bind].tempfactors = 5

prot.write('erk_feat.pdb')

prot.tempfactors = 0
prot[mask_pea_bind].tempfactors = 1
prot.write('actual_contact.pdb')


