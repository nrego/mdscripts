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
    parser.add_argument('--pred-contact', type=str, default='beta_phi_*/pred_contact_mask.dat', 
                        help='glob string for mask for predicted contacts, for each phi (default: %(default)s)')
    parser.add_argument('--buried-mask', type=str, default='../bound/buried_mask.dat',
                        help='mask for buried atoms (default: %(default)s')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot ROC, scores w/ phi')
    parser.add_argument('--no-beta', action='store_true', default=False,
                        help='If provided, assume input files are in kJ/mol, not kT')
    args = parser.parse_args()


    buried_mask = np.loadtxt(args.buried_mask, dtype=bool)
    surf_mask = ~buried_mask
    contact_mask = np.loadtxt(args.actual_contact, dtype=bool)
    pred_contacts = glob.glob(args.pred_contact)

    if contact_mask[surf_mask].sum() != contact_mask.sum():
        diff_contact = contact_mask.sum() - contact_mask[surf_mask].sum()
        print("WARNING: {:1d} Contacts are buried by buried_mask".format(diff_contact))
    contact_mask = contact_mask[surf_mask] # Only considering surface atoms

    print('Number of surface atoms: {}'.format(surf_mask.sum()))
    print('Number of contacts: {}'.format(contact_mask.sum()))


    header = 'beta*phi  tp   fp   tn   fn   tpr   fpr   prec   f_h   f_1   mcc'   
    dat = np.zeros((len(pred_contacts), 11))

    for i,fname in enumerate(pred_contacts):
        if args.no_beta:
            beta_phi = beta*float(os.path.dirname(fname).split('_')[-1]) / 10.0
        else:
            beta_phi = float(os.path.dirname(fname).split('_')[-1])/ 100.0

        pred_contact_mask = np.loadtxt(fname, dtype=bool)[surf_mask]

        tp = (pred_contact_mask & contact_mask).sum()
        fp = (pred_contact_mask & ~contact_mask).sum()
        tn = (~pred_contact_mask & ~contact_mask).sum()
        fn = (~pred_contact_mask & contact_mask).sum()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tnr = 1 - fpr
        assert np.isclose(tnr,  tn/(fp+tn))

        # precision, or positive predictive value
        ppv = tp/(tp+fp)
        if tp+fp == 0:
            ppv = 0.0

        # harmonic average of tpr and (1-fpr), or the tnr (aka specificity)
        f_h = harmonic_avg(tpr,tnr)
        
        # harmonic average of tpr and prec
        f_1 = harmonic_avg(tpr,ppv)

        # matthews correlation coef
        mcc = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(fp+tn)*(tn+fn))

        if tp+fp != 0:
            assert np.isclose(f_1, (2*tp)/(2*tp+fp+fn))
        else:
            f_1 = 0

        dat[i] = beta_phi, tp, fp, tn, fn, tpr, fpr, ppv, f_h, f_1, mcc


    # check that the data in sorted order by phi...
    sort_idx = np.argsort(dat[:,0])

    dat = dat[sort_idx]

    np.savetxt('performance.dat', dat, header=header, fmt='%1.2e')

    if args.plot:
        plt.plot(dat[:,6], dat[:,5], 'o')
        plt.show()
        plt.plot(dat[:,0], dat[:,-3], '-o', label=r'$f_h$')
        plt.plot(dat[:,0], dat[:,-2], '-o', label=r'$f_1$')
        plt.plot(dat[:,0], dat[:,-1], '-o', label='mcc')

        plt.legend()
        plt.show()
