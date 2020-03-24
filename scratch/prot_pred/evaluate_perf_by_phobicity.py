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
    args = parser.parse_args()


    buried_mask = np.loadtxt(args.buried_mask, dtype=bool)
    surf_mask = ~buried_mask
    contact_mask = np.loadtxt(args.actual_contact, dtype=bool)
    hydropathy_mask = np.loadtxt(args.hydropathy_mask, dtype=bool)
    
    rho_ds = np.load(args.rho_cross)

    # Beta phi vals, Shape: n_bphi_vals
    beta_phi_vals = rho_ds['beta_phi']
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


    print('Number of surface atoms: {}'.format(surf_mask.sum()))
    print('Number of contacts: {}'.format(contact_mask.sum()))
    print('Number of non-polar surface atoms: {} ({:0.2f})'.format(hydropathy_mask.sum(), (hydropathy_mask.sum()/surf_mask.sum())))
    print('Number of non-polar contact atoms: {} ({:0.2f})'.format((hydropathy_mask&contact_mask).sum(), (hydropathy_mask&contact_mask).sum() / contact_mask.sum()))

    header = 'beta*phi  tp(np) tp(p)  fp(np)  fp(p)  tn(np)  tn(p)  fn(np)  fn(p)'   
    dat = np.zeros((len(pred_contacts), 9))

    for i,fname in enumerate(pred_contacts):

        if args.no_beta:
            beta_phi = beta*float(os.path.dirname(fname).split('_')[-1]) / 10.0
        else:
            beta_phi = float(os.path.dirname(fname).split('_')[-1])/ 100.0

        pred_contact_mask = np.loadtxt(fname, dtype=bool)
        #assert pred_contact_mask[buried_mask].sum() == 0

        pred_contact_mask = pred_contact_mask[surf_mask]

        tp_np = (pred_contact_mask & contact_mask & hydropathy_mask).sum()
        fp_np = (pred_contact_mask & ~contact_mask & hydropathy_mask).sum()
        tn_np = (~pred_contact_mask & ~contact_mask & hydropathy_mask).sum()
        fn_np = (~pred_contact_mask & contact_mask & hydropathy_mask).sum()

        tp_po = (pred_contact_mask & contact_mask & ~hydropathy_mask).sum()
        fp_po = (pred_contact_mask & ~contact_mask & ~hydropathy_mask).sum()
        tn_po = (~pred_contact_mask & ~contact_mask & ~hydropathy_mask).sum()
        fn_po = (~pred_contact_mask & contact_mask & ~hydropathy_mask).sum()


        dat[i] = beta_phi, tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po


    # check that the data in sorted order by phi...
    sort_idx = np.argsort(dat[:,0])

    dat = dat[sort_idx]

    np.savetxt('perf_by_chemistry.dat', dat, header=header, fmt='%1.2e')

