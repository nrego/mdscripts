import os, glob
import numpy as np
from scratch.sam.util import *

## Extracts all data from patterned dataset (6 x 6, 4 x 9, or 4 x 4) ###

fnames = sorted(glob.glob('k_*/d_*/trial_0/PvN.dat')) + sorted(glob.glob('../inv_pattern_sample/l_*/d_*/trial_0/PvN.dat')) + sorted(glob.glob('k_*/PvN.dat'))
#fnames = sorted(glob.glob('*/d_*/trial_0/PvN.dat'))
p = 6
q = 6
#ds = np.load("sam_pattern_data.dat.npz")

#energies_old = ds['energies']
#methyl_pos_old = ds['methyl_pos']

#myfeat_old = np.zeros((len(energies_old), 3))

#for i, methyl_mask in enumerate(methyl_pos_old):
#    state = State(np.arange(36).astype(int)[methyl_mask])

#    myfeat_old[i] = state.k_o, state.n_oo, state.n_oe

#perf_mse_old, err, xvals, fit, reg_old = fit_leave_one(myfeat_old, energies_old)

myfeat_new = np.zeros((len(fnames), 3))
energies_new = np.zeros(len(fnames))
energies_new[:] = -1
energies_err = np.zeros_like(energies_new)

states = np.empty_like(energies_new, dtype=object)

for i, fname in enumerate(fnames):
    pvn = np.loadtxt(fname)[0]
    pt_idx = np.loadtxt('{}/this_pt.dat'.format(os.path.dirname(fname)), dtype=int)
    state = State(pt_idx, p, q)

    myfeat_new[i] = state.k_o, state.n_oo, state.n_oe
    energies_new[i] = pvn[1]
    energies_err[i] = pvn[2]
    states[i] = state

perf_mse_new, err, xvals, fit, reg_new = fit_leave_one(myfeat_new, energies_new)


np.savez_compressed('sam_pattern_{:02d}_{:02d}'.format(p, q), states=states, energies=energies_new, err_energies=energies_err) 


