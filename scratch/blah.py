import os, glob

dirnames = np.sort(glob.glob('*pattern_sample/*/d_*/trial_0'))
n_dat = dirnames.size + 2 # for k=0 and k=36

old_ds = np.load('pooled_pattern_sample/old_sam_pattern_data.dat.npz')
positions = old_ds['positions']

methyl_base = np.zeros(36, dtype=bool)

methyl_pos = np.zeros((n_dat, 36), dtype=bool)
k_vals = np.zeros(n_dat)
energies = np.zeros(n_dat)

for i, dirname in enumerate(dirnames):
    methyl_mask = methyl_base.copy()
    pt_pos = np.loadtxt('{}/this_pt.dat'.format(dirname), dtype=int)
    methyl_mask[pt_pos] = True
    k_ch3 = methyl_mask.sum()

    energy = np.loadtxt('{}/f_k_all.dat'.format(dirname))[-1]

    methyl_pos[i] = methyl_mask
    k_vals[i] = k_ch3
    energies[i] = energy

# k_00
energy = np.loadtxt('pattern_sample/k_00/f_k_all.dat')[-1]
energies[-2] = energy
k_vals[-2] = 0

# k_36
energy = np.loadtxt('pattern_sample/k_36/f_k_all.dat')[-1]
energies[-1] = energy
k_vals[-1] = 36
methyl_pos[-1][:] = True

np.savez_compressed('pooled_pattern_sample/sam_pattern_data.dat', energies=energies, positions=positions, k_vals=k_vals, methyl_pos=methyl_pos)
