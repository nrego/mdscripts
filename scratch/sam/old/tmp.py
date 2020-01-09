import os, glob
import numpy as np
from scratch.sam.util import *

fnames = sorted(glob.glob('k_*/d_*/trial_0/PvN.dat')) + sorted(glob.glob('../inv_pattern_sample/l_*/d_*/trial_0/PvN.dat')) + sorted(glob.glob('k_*/PvN.dat'))
ds = np.load("sam_pattern_data.dat.npz")

energies_old = ds['energies']
methyl_pos_old = ds['methyl_pos']

myfeat_old = np.zeros((len(energies_old), 3))

for i, methyl_mask in enumerate(methyl_pos_old):
    state = State(np.arange(36).astype(int)[methyl_mask])

    myfeat_old[i] = state.k_o, state.n_oo, state.n_oe

perf_mse_old, err, xvals, fit, reg_old = fit_leave_one(myfeat_old, energies_old)

myfeat_new = np.zeros((len(fnames), 3))
energies_new = np.zeros(len(fnames))
energies_new[:] = -1
energies_err = np.zeros_like(energies_new)

for i, fname in enumerate(fnames):
    pvn = np.loadtxt(fname)[0]
    pt_idx = np.loadtxt('{}/this_pt.dat'.format(os.path.dirname(fname)), dtype=int)
    state = State(pt_idx)

    myfeat_new[i] = state.k_o, state.n_oo, state.n_oe
    energies_new[i] = pvn[1]
    energies_err[i] = pvn[2]

perf_mse_new, err, xvals, fit, reg_new = fit_leave_one(myfeat_new, energies_new)


