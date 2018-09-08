from __future__ import division

import numpy as np
import MDAnalysis

import argparse


univ = MDAnalysis.Universe('cent_oh.gro')
cent_ids = np.loadtxt('cent_resids.dat').astype(int)
assert np.unique(cent_ids).size == cent_ids.size == 36

univ.add_TopologyAttr('tempfactor')
univ.atoms.tempfactors = 1

cent_group = univ.residues[cent_ids]
cent_group.atoms.tempfactors = 0

univ.atoms.write('blah.pdb')

meth_u = MDAnalysis.Universe('whole_ch3.gro')
oh_u = MDAnalysis.Universe('cent_oh.gro')

meth_residues = meth_u.residues[-cent_ids.size:]
oh_residues = oh_u.residues[cent_ids]

for i, res in enumerate(univ.residues[cent_ids]):
    ag_ref = res.atoms
    ag_meth = meth_residues[i].atoms
    ag_oh = oh_residues[i].atoms

    # shift vector from sulfur groups
    meth_shift = ag_ref[-1].position - ag_meth[-1].position
    oh_shift = ag_ref[-1].position - ag_oh[-1].position

    ag_meth.positions += meth_shift
    ag_oh.positions += oh_shift

newuniv = MDAnalysis.core.universe.Merge(univ.atoms[univ.atoms.tempfactors==1], oh_residues.atoms)

newuniv.atoms.write('new_center.gro')