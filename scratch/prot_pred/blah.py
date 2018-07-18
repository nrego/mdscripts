from __future__ import division, print_function

from constants import k
import os, glob
import numpy as np
import MDAnalysis

def get_water_nums_per_slab(water_positions, slab_pos, axis=0):
    bin_idx = np.digitize(water_positions[:,axis], bins=slab_pos) - 1

    ret_arr = np.zeros(slab_pos.size-1, dtype=float)

    for i in range(slab_pos.size-1):
        ret_arr[i] = (bin_idx == i).sum()

    return ret_arr


dx = 0.5
slab_x_pos = np.arange(0, 91, dx)
slab_vol = dx*70.*70. # in A^3
rho_0 = 0.033 # waters per A^3
univ = MDAnalysis.Universe('equil.gro', 'traj.xtc')

water_atoms = univ.select_atoms('name OW')

density_arr = np.zeros((univ.trajectory.n_frames, slab_x_pos.size-1), dtype=float)

for i_frame, ts in enumerate(univ.trajectory):
    water_pos = water_atoms.positions

    this_density = get_water_nums_per_slab(water_pos, slab_x_pos)

    density_arr[i_frame, :] = this_density

avg_density = density_arr.mean(axis=0)

out_arr = np.dstack((slab_x_pos[:-1], avg_density)).squeeze()

np.savetxt('avg_density.dat', out_arr)



def bootstrap(dat_all, n_uncorr_samples, fn):
    np.random.seed()
    boot_sample = np.random.choice(dat_all, n_uncorr_samples)

    return fn(boot_sample)

n_boot = 1000

boot_avg_ch3 = np.zeros(n_boot)
boot_var_ch3 = np.zeros(n_boot)
boot_avg_oh = np.zeros(n_boot)
boot_var_oh = np.zeros(n_boot)


for i_boot in xrange(n_boot):
    boot_avg_ch3[i_boot] = bootstrap(dat_ch3, 780, lambda x: x.mean())
    boot_avg_oh[i_boot] = bootstrap(dat_oh, 780, lambda x: x.mean())
    boot_var_ch3[i_boot] = bootstrap(dat_ch3, 780, lambda x: x.var(ddof=1))
    boot_var_oh[i_boot] = bootstrap(dat_oh, 780, lambda x: x.var(ddof=1))

