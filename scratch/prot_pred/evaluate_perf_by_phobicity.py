from __future__ import division, print_function

import glob, os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from mdtools import dr

from constants import k

import MDAnalysis

import argparse
from IPython import embed


def get_perf(dewet_mask, contact_mask, hydropathy_mask):

    tp_np = (dewet_mask & contact_mask & hydropathy_mask).sum()
    fp_np = (dewet_mask & ~contact_mask & hydropathy_mask).sum()
    tn_np = (~dewet_mask & ~contact_mask & hydropathy_mask).sum()
    fn_np = (~dewet_mask & contact_mask & hydropathy_mask).sum()

    tp_po = (dewet_mask & contact_mask & ~hydropathy_mask).sum()
    fp_po = (dewet_mask & ~contact_mask & ~hydropathy_mask).sum()
    tn_po = (~dewet_mask & ~contact_mask & ~hydropathy_mask).sum()
    fn_po = (~dewet_mask & contact_mask & ~hydropathy_mask).sum()


    return tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po


beta = 1/(300*k)

### A mix of evaluate perforance.py and anayze patch.py
#      Evaluates the fraction of TP, FP, etc that are non-polar or polar/charged

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluate performance with phi by atom chemistry')
    parser.add_argument('--actual-contact', type=str, default='../bound/actual_contact_mask.dat', 
                        help='mask for actual contacts (default: %(default)s)')
    parser.add_argument('--rho-cross', type=str, default='../reweight_data/rho_cross.dat.npz',
                        help='Dataset showing all atoms, including interpolated values at which each atom crosses thresh (default: %(default)s)')
    parser.add_argument('--buried-mask', type=str, default='beta_phi_000/buried_mask.dat',
                        help='mask for buried atoms (default: %(default)s')
    parser.add_argument('--hydropathy-mask', type=str, default='../bound/hydropathy_mask.dat',
                        help='mask of non-polar atoms (default: %(default)s)')
    parser.add_argument('--no-beta', action='store_true', default=False,
                        help='If provided, assume input files are in kJ/mol, not kT')
    parser.add_argument('--thresh', '-s', type=float, default=0.5,
                        help='Rho threshold for determining if atom is dewetted (default: %(default)s)')
    args = parser.parse_args()

    s = args.thresh
    buried_mask = np.loadtxt(args.buried_mask, dtype=bool)
    surf_mask = ~buried_mask
    contact_mask = np.loadtxt(args.actual_contact, dtype=bool)
    hydropathy_mask = np.loadtxt(args.hydropathy_mask, dtype=bool)
    
    rho_ds = np.load(args.rho_cross)

    # Beta phi vals, Shape: n_bphi_vals
    beta_phi_vals = np.round(rho_ds['beta_phi'], 4)
    # <rho_i>_phi, Shape: (n_heavy_atoms, n_bphi_vals)
    rho_dat = rho_ds['rho_data']
    # Critical bphis for each atom - odd number of bphis means atom i is dewetted by bphimax
    # Shape: (n_heavy_atoms)
    cross_vals = rho_ds['cross_vals']
    

    if contact_mask[surf_mask].sum() != contact_mask.sum():
        diff_contact = contact_mask.sum() - contact_mask[surf_mask].sum()
        print("WARNING: {:1d} Contacts are buried by buried_mask".format(diff_contact))
    
    contact_mask = contact_mask[surf_mask] # Only considering surface atoms
    hydropathy_mask = hydropathy_mask[surf_mask]
    cross_vals = cross_vals[surf_mask]
    rho_dat = rho_dat[surf_mask]
    cross_mask = np.diff(rho_dat < args.thresh)

    print('Number of surface atoms: {}'.format(surf_mask.sum()))
    print('Number of contacts: {}'.format(contact_mask.sum()))
    print('Number of non-polar surface atoms: {} ({:0.2f})'.format(hydropathy_mask.sum(), (hydropathy_mask.sum()/surf_mask.sum())))
    print('Number of non-polar contact atoms: {} ({:0.2f})'.format((hydropathy_mask&contact_mask).sum(), (hydropathy_mask&contact_mask).sum() / contact_mask.sum()))

    header = 'beta*phi  tp(np) tp(p)  fp(np)  fp(p)  tn(np)  tn(p)  fn(np)  fn(p)'   
    dat = []
    surf_indices = np.arange(surf_mask.sum())

    for i, beta_phi in enumerate(beta_phi_vals[:-1]):

        if args.no_beta:
            beta_phi = beta*beta_phi

        this_rho = rho_dat[:,i]
        next_rho = rho_dat[:,i+1]

        dewet_mask = this_rho < s

        ## Record state at this beta phi val

        tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po = get_perf(dewet_mask, contact_mask, hydropathy_mask)
        this_dat = beta_phi, tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po
        dat.append(this_dat)

        ## If we're about to cross a threshold for any atom(s), linearly interpolate their cross values
        this_cross_mask = cross_mask[:,i]
        n_cross = this_cross_mask.sum()

        if n_cross > 0:

            print("\n{} crosses from bphi={:.2f} to {:.2f}".format(n_cross, beta_phi, beta_phi_vals[i+1]))
            assert np.logical_xor((this_rho < s), (next_rho < s)).sum() == n_cross

            # Should always be the same, but oh well
            delta_bphi = np.round(beta_phi_vals[i+1] - beta_phi, 4)
            delta_rho = next_rho - this_rho
            slope = delta_rho / delta_bphi

            # Deltas giving the critical points for each atom
            d_bphi = ((s-this_rho)/(slope))[this_cross_mask]
            cross_indices = surf_indices[this_cross_mask]

            assert (d_bphi > 0).all()
            assert np.max(d_bphi) < delta_bphi

            sort_idx = np.argsort(d_bphi)

            for delta_index in sort_idx:
                this_idx = cross_indices[delta_index]
                this_d_bphi = d_bphi[delta_index]
                new_rho = this_rho + slope*this_d_bphi
                new_beta_phi = beta_phi + this_d_bphi
                dewet_mask = dewet_mask.copy()
                
                # This atom has dewetted
                if this_rho[this_idx] >= s:
                    assert next_rho[this_idx] < s
                    dewet_mask[this_idx] = True
                # This atom is wetting (flicker)
                elif this_rho[this_idx] < s:
                    assert next_rho[this_idx] >= s
                    dewet_mask[this_idx] = False
    
                tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po = get_perf(dewet_mask, contact_mask, hydropathy_mask)
                this_dat = new_beta_phi, tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po
                dat.append(this_dat)

    dat = np.array(dat)
    # check that the data in sorted order by phi...
    assert np.diff(dat[:,0]).min() >= 0
    beta_phi_vals, tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po = [arr.squeeze() for arr in np.split(dat, dat.shape[1], axis=1)]
    n_dewet = tp_np + tp_po + fp_np + fp_po

    assert np.abs(np.diff(n_dewet)).max() <= 1

    np.savetxt('perf_by_chemistry.dat', dat, header=header, fmt='%1.8e')

