import os, glob
import numpy as np
from scratch.sam.util import *

fnames = sorted(glob.glob('*.dat'))

def get_energy(pt_idx, methyl_mask, nn, ext_count, reg):
    coef1, coef2, coef3 = reg.coef_
    inter = reg.intercept_

    this_pt_o = np.setdiff1d(np.arange(36), pt_idx)

    k_oh = 36 - methyl_mask.sum()
    oo_ext = 0
    mo_int = 0
    for o_idx in this_pt_o:
        oo_ext += ext_count[o_idx]
    for m_idx in pt_idx:
        for n_idx in nn[m_idx]:
            mo_int += ~methyl_mask[n_idx]

    return inter + coef1*k_oh + coef2*mo_int + coef3*oo_ext


ds = np.load('../../pooled_pattern_sample/m3.dat.npz')
perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(ds['feat_vec'], ds['energies'], do_ridge=False)

for fname in fnames:
    dirname = fname.split('.')[0]
    pvn = np.loadtxt('{}/PvN.dat'.format(dirname))[0,1]
    methyl_mask = np.loadtxt(fname, dtype=bool)
    pt_idx = np.arange(36)[methyl_mask]

    state = State(pt_idx, reg=reg, e_func=get_energy)
    print(fname)
    print("  actual f: {:0.2f}".format(pvn))
    print("  estimated f: {:0.2f}".format(state.energy))
    print("\n")
    plt.close('all')
    state.plot()
    plt.savefig('/Users/nickrego/Desktop/{}.png'.format(dirname))
    