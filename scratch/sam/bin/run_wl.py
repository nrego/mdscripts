import numpy as np

import argparse
import os, glob

from scipy.spatial import cKDTree

from scratch.sam.util import *
from wang_landau import WangLandau


def simple_ising_energy(pt_idx, methyl_mask, nn, coef1, coef2, inter):
    k_ch3 = methyl_mask.sum()
    n_mm = 0
    for i in pt_idx:
        for j in nn[i]:
            if j>i and methyl_mask[j]:
                n_mm += 1

    return inter + coef1*k_ch3 + coef2*n_mm

# On N_m, n_mm, and n_mo (internal)
def complex_ising_energy(pt_idx, methyl_mask, nn, coef1, coef2, coef3, inter):
    k_ch3 = methyl_mask.sum()
    n_mm = 0
    n_mo = 0
    for i in pt_idx:
        for j in nn[i]:
            if j>i and methyl_mask[j]:
                n_mm += 1
            elif not methyl_mask[j]:
                n_mo += 1

    return inter + coef1*k_ch3 + coef2*n_mm + coef3*n_mo

### Extract data ###

print("\nLoading data...")
energies, methyl_pos, k_vals, positions, pos_ext, patch_indices, nn, nn_ext, edges, ext_indices, int_indices = extract_data()
n_pts = energies.size

print("  ...Done. Loaded {} patterns.".format(n_pts))

assert n_pts == methyl_pos.shape[0]

# mm, oo, mo, mo_ext, oo_ext
k_eff_all = np.zeros((n_pts, 5), dtype=int)

print("\nAssembling counts of edge types for each pattern...")
for i in range(n_pts):
    methyl_mask = methyl_pos[i]
    this_k_eff = get_keff_all(methyl_mask, edges, patch_indices)

    assert np.array_equal(this_k_eff.sum(axis=0), this_k_eff[int_indices].sum(axis=0) + this_k_eff[ext_indices].sum(axis=0))

    k_eff_all[i] = this_k_eff.sum(axis=0)

print("  ...Done.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Find entropy (density of states) of energy for a fixed k_ch3 \
                                      Using an ising-like model')
    parser.add_argument('--k-ch3', default=1, type=int,
                        help='k_ch3 to find density of states (default: %(default)s)')
    parser.add_argument('--do-brute', action='store_true',
                        help='If true, get density of states by exhaustively generating each configuration at this \
                              k_ch3 (default: only do brute if k_ch3 <= 6 or k_ch3 >= 30')
    parser.add_argument('--de', default=0.5, type=float,
                        help='Bin width for energy, in kT (default: %(default)s)')
    parser.add_argument('--eps', type=float, default='1e-10',
                        help='Epsilon (tolerance) for Wang Landau (default: 1e-10)')
    parser.add_argument('--max-wl-iter', type=int, default=60000,
                        help='Maximum number of WL MC steps to take each iteration (default: %(default)s)')
    parser.add_argument('--hist-flat-tol', type=float, default=0.8,
                        help='Criterion for determining if histogram is flat for each \
                              Wang-Landau iteration (default: %(default)s)')
    parser.add_argument('--patch-size', type=int, default=6,
                        help='size of patch, in groups on each side (default: %(default)s)')



    args = parser.parse_args()

    ## Simple ising model - N_m and n_mm
    feat_vec = np.vstack((k_vals, k_eff_all[:,0])).T

    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies)

    min_val = reg.coef_[0] * 36 + reg.coef_[1] * 85
    # max val is 0, by defn

    ## Set up and run WL
    bins = np.arange(0, 155, 0.5)
    fn_kwargs = dict(nn=nn, coef1=reg.coef_[0], coef2=reg.coef_[1], inter=-min_val)

    wl = WangLandau(positions, bins, simple_ising_energy, fn_kwargs=fn_kwargs, eps=args.eps, max_iter=args.max_wl_iter)

    wl.gen_states(args.k_ch3, args.do_brute, hist_flat_tol=args.hist_flat_tol)

    np.savez_compressed('m2_density_k_c_{:03d}.dat'.format(args.k_ch3), bins=bins, entropies=wl.entropies, density=wl.density, k_ch3=args.k_ch3, eps=args.eps, hist_flat_tol=args.hist_flat_tol,
                        do_brute=args.do_brute, omega=wl.omega, sampled_pt_idx=wl.sampled_pt_idx)


    # Part 2 - more complex model - N_m, n_mm, and n_mo_int
    feat_vec = np.vstack((k_vals, k_eff_all[:,0], k_eff_all[:,2])).T
    perf_r2, perf_mse, err, xvals, fit, reg = fit_general_linear_model(feat_vec, energies)
    min_val = reg.coef_[0] * 36 + reg.coef_[1] * 85 + reg.coef_[0] * 0

    #bins = np.arange(0, 155, 0.5)
    bins = np.arange(135, 286.5, 0.5)

    fn_kwargs = dict(nn=nn, coef1=reg.coef_[0], coef2=reg.coef_[1], coef3=reg.coef_[2], inter=reg.intercept_)

    wl = WangLandau(positions, bins, complex_ising_energy, fn_kwargs=fn_kwargs, eps=args.eps, max_iter=args.max_wl_iter)
    wl.gen_states(args.k_ch3, args.do_brute, hist_flat_tol=args.hist_flat_tol)

    np.savez_compressed('m3_density_k_c_{:03d}.dat'.format(args.k_ch3), bins=bins, entropies=wl.entropies, density=wl.density, k_ch3=args.k_ch3, eps=args.eps, hist_flat_tol=args.hist_flat_tol,
                        do_brute=args.do_brute, omega=wl.omega, sampled_pt_idx=wl.sampled_pt_idx)


