import numpy as np
import os, glob

from constants import k

## Run from trial_0 directory
fnames = sorted(glob.glob('phi_*/rho_data_dump_rad_6.0.dat.npz'))
dirnames = sorted(glob.glob('phi_*'))

assert len(fnames) == len(dirnames), "Error: missing some rho_data dumps"

n_files = len(fnames)

phi_vals = np.array([float(dname.split('_')[1]) / 10.0 for dname in dirnames])

# Shape: (36, n_frames)
n_0 = np.load(fnames[0])['rho_water'][30:, :].T.mean(axis=1)

avg_nis2 = np.empty((36, n_files))

for i, fname in enumerate(fnames):
    n_phi = np.load(fname)['rho_water'][30:, :].T.mean(axis=1)

    avg_nis2[:, i] = n_phi