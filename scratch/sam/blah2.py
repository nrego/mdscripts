from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed

def generate_pattern(univ, oh_univ, ids_res):
    univ.atoms.tempfactors = 1

    for idx_res in ids_res:
        res = univ.residues[idx_res]
        ag_ref = res.atoms
        res.atoms.tempfactors = 0

        oh_res = oh_univ.residues[idx_res]
        ag_oh = oh_res.atoms

        oh_shift = ag_ref[-1].position - ag_oh[-1].position
        ag_oh.positions += oh_shift

    newuniv = MDAnalysis.core.universe.Merge(univ.atoms[univ.atoms.tempfactors == 1], oh_univ.residues[ids_res].atoms)

    univ.atoms.tempfactors = 1

    return newuniv

univ = MDAnalysis.Universe('../whole_ch3.gro')

univ.add_TopologyAttr('tempfactor')
univ.atoms.tempfactors = 1

cent_group = univ.residues[-36:]
cent_ids = cent_group.resids - 1
cent_group.atoms.tempfactors = 0

#univ.atoms.write('blah.pdb')

meth_u = MDAnalysis.Universe('../whole_ch3.gro')
oh_u = MDAnalysis.Universe('../cent_oh.gro')

meth_residues = meth_u.residues[-36:]
oh_residues = oh_u.residues[-36:]

# number of OH groups
n_group = 9
# Total number of patterns
n_batch = 1
patterns = []

i_pattern = 0

head_cg = univ.residues[-36:].atoms.select_atoms('resname CH3 and name CT').center_of_geometry()
while len(patterns) < n_batch:

    this_choice = sorted(np.random.choice(cent_ids, size=n_group, replace=False))
    if this_choice in patterns:
        continue

    newuniv = generate_pattern(univ, oh_u, this_choice)
    head_groups = newuniv.residues[-36:].atoms.select_atoms('resname OH and type O')

    sq_dist = np.sum((head_groups.positions - head_groups.center_of_geometry())**2, axis=1)
    rms = np.sqrt(sq_dist.mean())
    centroid_dist = np.linalg.norm(head_groups.center_of_geometry() - head_cg)
    if rms < 13:
        continue

    
    patterns.append(this_choice)
    print('pos {:03d}: rms: {:.1f} cog_dist: {:.1f}'.format(i_pattern, rms, centroid_dist))

    newuniv.atoms.write('pos{:03d}.gro'.format(i_pattern))
    i_pattern += 1

patterns = np.array(patterns)


np.savetxt('patterns.dat', patterns)
