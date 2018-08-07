# Analyze results of dynamic_indus (i.e. per-atom, time resolved number of waters and other prot atoms)
#   to locate hydrophobic patches on protein
from __future__ import division, print_function

import MDAnalysis
import os, glob
import numpy as np
import matplotlib 
from matplotlib import cm
from matplotlib import pyplot as plt

import numpy as np

import argparse

from util import find_dewet_atoms, compare_pred_actual_acc
from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument('--buried-mask', default='buried_mask.dat')
parser.add_argument('--contact-mask', default='contact_mask.dat', help='boolean arr of actual contacts')
parser.add_argument('--target-struct', help='pdb file of dynamic_water_avg for target state', type=str)
parser.add_argument('--rho-ref', help='rho_data file for unbound, unbiased ref')
parser.add_argument('--rho-targ', help='rho_data file for target')
parser.add_argument('--outprefix', type=str, default='', help='prefix for output files')
parser.add_argument('--thresh', type=float, help='If provided, use thresh as cutoff for determining dewet atoms (predictions). otherwise, make roc.')


args = parser.parse_args()
thresh = args.thresh
outpref = args.outprefix

contact_mask_actual = np.loadtxt(args.contact_mask).astype(bool)
buried_mask = np.loadtxt(args.buried_mask).astype(bool)
surf_mask = ~buried_mask
n_tot = surf_mask.sum() # total number of surface atoms - candidates
target_univ = MDAnalysis.Universe(args.target_struct)

target_avg_water = np.load(args.rho_targ)['rho_water'].mean(axis=0)
ref_avg_water = np.load(args.rho_ref)['rho_water'].mean(axis=0)

assert target_avg_water.size == ref_avg_water.size == target_univ.atoms.n_atoms == contact_mask_actual.size == buried_mask.size


n_actual = contact_mask_actual[surf_mask].sum()


if thresh is not None:
    normed_rho, this_pred_contacts = find_dewet_atoms(ref_avg_water, target_avg_water, buried_mask, dewetting_thresh=thresh)

    tp, fp, tn, fn = compare_pred_actual_acc(this_pred_contacts[surf_mask], contact_mask_actual[surf_mask])

    n_pred = this_pred_contacts[surf_mask].sum()

    np_tot = tp + fn
    nn_tot = tn + fp

    assert np_tot + nn_tot == n_tot

    tpr = tp/np_tot
    fpr = fp/nn_tot
    acc = (tp+tn)/n_tot

    print("thresh: {}  n_pred: {}  n_actual: {}  n_surf: {}  TPR: {}  FPR: {}  ACC: {}".format(thresh, n_pred, n_actual, n_tot, tpr, fpr, acc))
    print("  TP: {}  FP: {}  TN: {}  FN: {}".format(tp, fp, tn, fn))
    target_univ.atoms.tempfactors = 1
    target_univ.atoms[this_pred_contacts].tempfactors = 0
    target_univ.atoms.write('{}pred_contacts.pdb'.format(outpref)) 

# No threshold provided, cycle through and produce ROC curve
if thresh is None:
    ideal_pt = np.array([0,1.])
    # Cycle thru thresholds and determine bound atoms
    thresholds = np.arange(0, 1.05, 0.05)

    # FPR, TPR
    roc = np.zeros((thresholds.size, 3))
    roc[:,0] = thresholds

    for i, thresh in enumerate(thresholds):
        normed_rho, this_pred_contacts = find_dewet_atoms(ref_avg_water, target_avg_water, buried_mask, dewetting_thresh=thresh)

        tp, fp, tn, fn = compare_pred_actual_acc(this_pred_contacts[surf_mask], contact_mask_actual[surf_mask])

        n_pred = this_pred_contacts[surf_mask].sum()

        np_tot = tp + fn
        nn_tot = tn + fp

        assert np_tot + nn_tot == n_tot

        tpr = tp/np_tot
        fpr = fp/nn_tot
        acc = (tp+tn)/n_tot

        roc[i, 1] = fpr
        roc[i, 2] = tpr

        dist = np.sqrt(fpr**2 + (1-tpr)**2)


        print("thresh: {}  n_pred: {}  n_actual: {}  n_surf: {}  TPR: {:.4f}  FPR: {:.4f}  ACC: {:.4f}  D: {:.4f}".format(thresh, n_pred, n_actual, n_tot, tpr, fpr, acc, dist))

    plt.plot(roc[:,0], roc[:,1], '-o')
    #plt.show()

    np.savetxt('{}roc.dat'.format(outpref), roc, header='thresh    TPR     FPR')


