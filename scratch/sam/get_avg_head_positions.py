
u_oh = MDAnalysis.Universe('top_00.tpr', 'traj_cent_000.xtc')
u_ch3 = MDAnalysis.Universe('top_36.tpr', 'traj_cent_036.xtc')

s_oh = u_oh.select_atoms('name S1 or name S')
s_ch3 = u_ch3.select_atoms('name S1 or name S')

head_oh = u_oh.select_atoms('name CT or name O12')
head_ch3 = u_ch3.select_atoms('name CT or name O12')


pos_s_oh = np.zeros((u_oh.trajectory.n_frames, s_oh.n_atoms, 3))
pos_s_ch3 = np.zeros_like(pos_s_oh)
pos_head_oh = np.zeros_like(pos_s_oh)
pos_head_ch3 = np.zeros_like(pos_s_oh)

for i in range(u_oh.trajectory.n_frames):
    u_oh.trajectory[i]
    u_ch3.trajectory[i]

    pos_s_oh[i, ...] = s_oh.positions
    pos_s_ch3[i, ...] = s_ch3.positions
    pos_head_oh[i, ...] = head_oh.positions
    pos_head_ch3[i, ...] = head_ch3.positions

avg_pos_s_oh = pos_s_oh.mean(axis=0)
avg_pos_s_ch3 = pos_s_ch3.mean(axis=0)
avg_pos_head_oh = pos_head_oh.mean(axis=0)
avg_pos_head_ch3 = pos_head_ch3.mean(axis=0)


plt.plot(avg_pos_head_ch3[:-36, 1], avg_pos_head_ch3[:-36, 2], 'ko')
plt.plot(avg_pos_head_ch3[-36:, 1], avg_pos_head_ch3[-36:, 2], 'ro')

