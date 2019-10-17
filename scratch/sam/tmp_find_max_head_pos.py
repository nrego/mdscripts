from scipy.spatial import cKDTree

univ_ch3 = MDAnalysis.Universe('../prep_small/P5/whole.gro', '../prep_small/P5/whole.xtc')
#univ_ch3 = MDAnalysis.Universe('equil_ch3/equil.tpr', 'equil_ch3/whole.xtc')
univ_oh = MDAnalysis.Universe('equil_oh/equil.tpr', 'equil_oh/whole.xtc')

head_ch3 = univ_ch3.select_atoms("name CT")
head_oh = univ_oh.residues[-36:].atoms.select_atoms("name O12")

tree = cKDTree(head_oh.positions)
d, indices = tree.query(head_ch3.positions)

head_oh = head_oh[indices]

pos_head_ch3 = np.zeros((univ_oh.trajectory.n_frames, head_ch3.n_atoms, 3))
pos_head_oh = np.zeros((univ_oh.trajectory.n_frames, head_oh.n_atoms, 3))


for i in range(univ_oh.trajectory.n_frames):
    univ_ch3.trajectory[i]
    univ_oh.trajectory[i]

    pos_head_ch3[i, ...] = head_ch3.positions
    pos_head_oh[i, ...] = head_oh.positions

max_ch3 = pos_head_ch3.max(axis=(0,1))
max_oh = pos_head_oh.max(axis=(0,1))

min_ch3 = pos_head_ch3.min(axis=(0,1))
min_oh = pos_head_oh.min(axis=(0,1))
