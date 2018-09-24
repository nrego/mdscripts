from __future__ import division

import numpy as np
import MDAnalysis

import argparse

univ_ch3 = MDAnalysis.Universe('equil_ch3/equil.tpr', 'equil_ch3/whole.xtc')
univ_oh = MDAnalysis.Universe('equil_oh/equil.tpr', 'equil_oh/whole.xtc')

assert univ_ch3.trajectory.n_frames == univ_oh.trajectory.n_frames

res_ch3 = univ_ch3.residues[-36:]
res_oh = univ_oh.residues[-36:]

head_ch3 = res_ch3.atoms.select_atoms('type CT')
head_oh = res_oh.atoms.select_atoms('type OH')

assert head_ch3.n_atoms == head_oh.n_atoms == 36

min_pt = np.array([np.inf, np.inf, np.inf])
max_pt = - min_pt


for i in range(univ_ch3.trajectory.n_frames):
    univ_ch3.trajectory[i]
    univ_oh.trajectory[i]

    if head_ch3.positions[:,0].min() < 16 or head_oh.positions[:,0].min() < 16:
        break

    max_pt = np.max([max_pt, head_ch3.positions.max(axis=0), head_oh.positions.max(axis=0)], axis=0)
    min_pt = np.min([min_pt, head_ch3.positions.min(axis=0), head_oh.positions.min(axis=0)], axis=0)

max_pt = np.ceil(max_pt)
min_pt = np.floor(min_pt)