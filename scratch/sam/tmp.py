import os, glob
import numpy as np
from scratch.sam.util import *

fnames = sorted(glob.glob('pattern_sample/k_*/d_*/trial_0/PvN.dat'))

n_dat = len(fnames)

methyl_pos = np.zeros((n_dat, 36), dtype=bool)
k_ch3 = np.zeros(n_dat)
f = np.zeros(n_dat)
f_old = np.zeros(n_dat)

for i, fname in enumerate(fnames):
    old_fname = 'bkup_{}'.format(fname)
    path = os.path.dirname(fname)
    old_path = 'bkup_{}'.format(path)

    pvn = np.loadtxt(fname)
    old_pvn = np.loadtxt(old_fname)

    pt = np.loadtxt('{}/this_pt.dat'.format(path)).astype(int)
    old_pt = np.loadtxt('{}/this_pt.dat'.format(old_path)).astype(int)

    assert np.array_equal(pt, old_pt)
    methyl_mask = np.zeros(36, dtype=bool)
    methyl_mask[pt] = True


    k_ch3[i] = methyl_mask.sum()
    methyl_pos[i] = methyl_mask
    f[i] = pvn[0][1]
    f_old[i] = old_pvn[0][1]

feat_vec = k_ch3[:,None]
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, f_old)

