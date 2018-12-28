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

    parser = argparse.ArgumentParser('Evaluate performance with phi')
    parser.add_argument('--actual-contact', type=str, default='../bound/actual_contact_mask.dat', 
                        help='mask for actual contacts (default: %(default)s)')
    parser.add_argument('--pred-contact', type=str, default='phi_*/pred_contact_mask.dat', 
                        help='glob string for mask for predicted contacts, for each phi (default: %(default)s)')
    parser.add_argument('--buried-mask', type=str, default='../bound/buried_mask.dat',
                        help='mask for buried atoms (default: %(default)s')
    args = parser.parse_args()


    buried_mask = np.loadtxt(args.buried_mask, dtype=bool)
    surf_mask = ~buried_mask
    contact_mask = np.loadtxt(args.actual_contact, dtype=bool)
    pred_contacts = np.sort(glob.glob(args.pred_contact))

    assert contact_mask[surf_mask].sum() == contact_mask.sum()
    contact_mask = contact_mask[surf_mask] # Only considering surface atoms

    print('Number of surface atoms: {}'.format(surf_mask.sum()))
    print('Number contacts: {}'.format(contact_mask.sum()))


    header = 'phi  tp   fp   tn   fn   tpr   fpr   prec   f_h   f_1   mcc'   
    dat = np.zeros((len(pred_contacts), 11))

    indices = np.arange(surf_mask.sum())

    fig, ax = plt.subplots(figsize=(100,10))
    for i,fname in enumerate(pred_contacts):
        
        phi = float(os.path.dirname(fname).split('_')[-1]) / 10.0

        this_phi = np.ones_like(indices).astype(float)
        this_phi[:] = beta*phi
        if phi == 0:
            pred_contact_mask = np.zeros_like(contact_mask)
        else:
            pred_contact_mask = np.loadtxt(fname, dtype=bool)[surf_mask]

        tp_mask = pred_contact_mask & contact_mask
        fp_mask = pred_contact_mask & ~contact_mask
        tn_mask = ~pred_contact_mask & ~contact_mask
        fn_mask = ~pred_contact_mask & contact_mask

        tp = (pred_contact_mask & contact_mask).sum()
        fp = (pred_contact_mask & ~contact_mask).sum()
        tn = (~pred_contact_mask & ~contact_mask).sum()
        fn = (~pred_contact_mask & contact_mask).sum()

        ax.plot(indices[tp_mask], this_phi[tp_mask], 's', color='#FF007F', markersize=2)
        ax.plot(indices[fp_mask], this_phi[fp_mask], 's', color='#FF7F00', markersize=2)
        ax.plot(indices[tn_mask], this_phi[tn_mask], 's', color='#3F3F3F', markersize=2)
        ax.plot(indices[fn_mask], this_phi[fn_mask], 's', color='#7F00FF', markersize=2)

    #plt.savefig('/home/nick/Desktop/dot_plot.svg')
    ax.set_xlim(-10,1400)
    #ax.set_ylim(0,4)
    fig.tight_layout()
    fig.savefig('/home/nick/Desktop/dot_plot.pdf')
