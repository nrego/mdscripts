from scipy.spatial import cKDTree

#univ_ch3 = MDAnalysis.Universe('P9_4/whole.gro', 'P9_4/traj.xtc')
univ_ch3 = MDAnalysis.Universe('whole.gro', 'whole.xtc')
univ_oh = MDAnalysis.Universe('../whole.gro', '../whole.xtc')
#univ_oh = MDAnalysis.Universe('equil_oh/whole.gro', 'equil_oh/whole.xtc')

head_ch3 = univ_ch3.select_atoms("name CT")
head_oh = univ_oh.select_atoms("name O12")
s_ch3 = univ_ch3.select_atoms('name S')
s_oh = univ_oh.select_atoms('name S1')

tree = cKDTree(s_oh.positions[:,1:])
d, indices = tree.query(s_ch3.positions[:,1:])

assert indices.size == np.unique(indices).size == head_ch3.n_atoms

head_oh = head_oh[indices]
s_oh = s_oh[indices]
assert head_oh.n_atoms == head_ch3.n_atoms

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
