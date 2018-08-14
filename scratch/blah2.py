from __future__ import division

import numpy as np
import MDAnalysis


univ = MDAnalysis.Universe('em_final/cent.gro')
cent_ids = np.loadtxt('cent_resids.dat', dtype=int) + 13
uniq_ids = np.unique(cent_ids)

univ.add_TopologyAttr('tempfactor')
univ.atoms.tempfactors = 1

cent_group = univ.residues[uniq_ids]
cent_group.atoms.tempfactors = 0

univ.atoms.write('blah.pdb')

meth_u = MDAnalysis.Universe("../../CH3_Disks/equil.gro")


meth_ag = meth_u.residues[:uniq_ids.size]

for i, res in enumerate(univ.residues[uniq_ids]):
    ag_ref = res.atoms[1:]
    ag_meth = meth_ag[i].atoms[3:]
    shift = res.atoms[1].position - meth_ag[i].atoms[3].position
    ag_meth.positions = ag_ref.positions
    meth_ag[i].atoms[:3].positions += shift

newuniv = MDAnalysis.core.universe.Merge(univ.atoms[univ.atoms.tempfactors==1], meth_ag.atoms)