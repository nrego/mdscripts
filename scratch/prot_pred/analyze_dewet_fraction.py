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

def harmonic_avg(*args):
    recip_avg = (1/np.array(args)).mean()

    return 1/recip_avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Analyze whether dewetted surf atoms are polar or non-polar')
    parser.add_argument('--actual-contact', type=str, default='../bound/actual_contact_mask.dat', 
                        help='mask for actual contacts (default: %(default)s)')
    parser.add_argument('--pred-contact', type=str, default='beta_phi_*/pred_contact_mask.dat', 
                        help='glob string for mask for predicted contacts, for each phi (default: %(default)s)')
    parser.add_argument('--buried-mask', type=str, default='../bound/buried_mask.dat',
                        help='mask for buried atoms (default: %(default)s')
    parser.add_argument('--hydropathy', type=str, default='../bound/hydropathy_mask.dat',
                        help='Hydropathy mask (Default: %(default)s)')
    parser.add_argument('--no-beta', action='store_true', default=False,
                        help='If provided, assume input files are in kJ/mol, not kT')
    args = parser.parse_args()


    contact_mask = np.loadtxt(args.actual_contact, dtype=bool)
    buried_mask = np.loadtxt(args.buried_mask, dtype=bool)
    surf_mask = ~buried_mask
    n_surf = surf_mask.sum()

    assert contact_mask[buried_mask].sum() == 0
    hydropathy_mask = np.loadtxt(args.hydropathy, dtype=bool)

    pred_contacts = sorted(glob.glob(args.pred_contact))

    if args.no_beta:
        bphis = np.array([beta*float(os.path.dirname(fname).split('_')[-1]) / 10.0 for fname in pred_contacts])
    else:
        bphis = np.array([float(os.path.dirname(fname).split('_')[-1])/ 100.0 for fname in pred_contacts])

    assert np.diff(bphis).min() > 0
    assert bphis.size == len(pred_contacts)

    outdat = np.zeros((bphis.size, 5))

    # Note: n_dewet_np + n_dewet_po + n_wet_np + n_wet_po = n_surf heavies
    headerstr = 'beta_phi     n_dewet_np    n_dewet_po   n_wet_np    n_wet_po'

    for i,fname in enumerate(pred_contacts):
        beta_phi = bphis[i]

        pred_mask = np.loadtxt(fname, dtype=bool)
        assert pred_mask[buried_mask].sum() == 0


        ## number of dewetted np surface atoms
        dewet_np = pred_mask[surf_mask] & hydropathy_mask[surf_mask]
        dewet_po = pred_mask[surf_mask] & ~hydropathy_mask[surf_mask]
        wet_np = ~pred_mask[surf_mask] & hydropathy_mask[surf_mask]
        wet_po = ~pred_mask[surf_mask] & ~hydropathy_mask[surf_mask]


        outdat[i] = beta_phi, dewet_np.sum(), dewet_po.sum(), wet_np.sum(), wet_po.sum()

        assert outdat[i,1:].sum() == n_surf

