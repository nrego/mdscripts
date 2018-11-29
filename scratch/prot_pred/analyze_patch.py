from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed
from mdtools import MDSystem
import cPickle as pickle
import argparse



def rms(pts, centroid):
    
    diff = pts-centroid
    diff_sq = diff**2
    
    return np.sqrt(diff_sq.sum(axis=1).mean())

def bootstrap(atm_grp, n_sub, n_boot=100):
    assert n_sub <= atm_grp.n_atoms

    np.random.seed()

    boot = np.zeros(n_boot)

    cog = atm_grp.center_of_geometry()

    for i in range(n_boot):
        boot_idx = np.random.choice(atm_grp.n_atoms, n_sub, replace=False)
        boot_rms = rms(atm_grp.positions[boot_idx], cog)

        boot[i] = boot_rms

    return boot.mean(), boot.std(ddof=1)


## Analyze composition of classified predictions

parser = argparse.ArgumentParser('analyze atomic composition of TPs, FPs, etc')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='Topology (TPR) file (unbound form!)')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Structure (GRO) file')
parser.add_argument('--sel-spec', type=str, default='segid seg_0_Protein_targ and not name H*',
                    help='Selection spec to get target heavy atoms')
parser.add_argument('--actual-contact', type=str, required=True,
                    help='Actual contacts')
parser.add_argument('--buried', type=str, default='buried_mask.dat',
                    help='Buried mask')
parser.add_argument('--pred-contact', type=str, required=True,
                    help='Predicted contact mask')
parser.add_argument('--hydropathy', type=str, required=True,
                    help='Hydropathy mask')

args = parser.parse_args()

buried_mask = np.loadtxt(args.buried, dtype=bool)
surf_mask = ~buried_mask

univ = MDAnalysis.Universe(args.top, args.struct)
targ = univ.select_atoms(args.sel_spec)[surf_mask]

contact_mask = np.loadtxt(args.actual_contact, dtype=bool)[surf_mask]
pred_mask = np.loadtxt(args.pred_contact, dtype=bool)[surf_mask]
hydropathy = np.loadtxt(args.hydropathy, dtype=bool)[surf_mask]

assert contact_mask.size == targ.n_atoms

# size of prediction
n_surf = targ.n_atoms

tp_mask = contact_mask & pred_mask
fp_mask = ~contact_mask & pred_mask
tn_mask = ~contact_mask & ~pred_mask
fn_mask = contact_mask & ~pred_mask

tp = tp_mask.sum()
fp = fp_mask.sum()
tn = tn_mask.sum()
fn = fn_mask.sum()

cond_pos = tp + fn
cond_neg = tn + fp

assert n_surf == cond_pos + cond_neg

prev = cond_pos / n_surf
prec = tp/(tp+fp)
tpr = tp/(cond_pos)
fpr = fp/(cond_neg)

mcc = ((tp*tn)-(fp*fn)) / np.sqrt((tp+fp)*(fp+tn)*(fp+fn)*(tn+fn))

# f1's limiting value as entire surface is predicted
f1_lim = 2/(1+ (1/prev))
f1 = (2*tp)/(2*tp+fp+fn)
d_h = 2/((1/tpr) + (1/(1-fpr)))

print("RESULTS")
print("_______")
print("")
print("n_surf: {}  P: {}  N: {}  prevalence (P/n_surf): {:.2f}".format(n_surf, cond_pos, cond_neg, prev))
print("TP: {}  FP: {}  TN: {}  FN: {}".format(tp, fp, tn, fn))
print("TPR (recall): {:0.2f}  FPR: {:0.2f}  1-FPR (specificity): {:0.2f}   1-TPR (miss rate): {:0.2f}".format(tpr, fpr, 1-fpr, 1-tpr))
print("Precision: {:0.2f}  F1: {:.2f}  (lim: {:.2f})  d_h: {:.2f}   MCC: {:.2f}".format(prec, f1, f1_lim, d_h, mcc))


print("...Analyzing composition of prediction groups...")


# Find fraction of groups that are phobic
#   Expectation: TP will be more hydrophobic than FNs (indicating that we are capturing the **hydrophobic** parts of patch)
tp_phob = (tp_mask & hydropathy).sum()/tp
fp_phob = (fp_mask & hydropathy).sum()/fp
tn_phob = (tn_mask & hydropathy).sum()/tn
fn_phob = (fn_mask & hydropathy).sum()/fn

print("Fraction hydrophobic:")
print("   TP: {:0.2f}  FN:  {:0.2f}  FP:  {:0.2f}  TN: {:0.2f}".format(tp_phob, fn_phob, fp_phob, tn_phob))


## Find the spread of the tp, fp, fn
#  Expectation: FN's around edge and non-continuous, so will have larger rms than TP's (which should be tighter cluster)
p_atoms = targ[contact_mask]
tp_atoms = targ[tp_mask]
fp_atoms = targ[fp_mask]
fn_atoms = targ[fn_mask]

cog = p_atoms.center_of_geometry()
p_spread = rms(p_atoms.positions, cog)
tp_spread = rms(tp_atoms.positions, cog)
fn_spread = rms(fn_atoms.positions, cog)
#fp_spread = rms(fp_atoms.positions)


# What's expected rms for a subpatch of P with n_TP atoms?
tp_expected_avg, tp_expected_se = bootstrap(p_atoms, tp)
fn_expected_avg, fn_expected_se = bootstrap(p_atoms, fn)

z_tp = (tp_spread - tp_expected_avg) / tp_expected_se
z_fn = (fn_spread - fn_expected_avg) / fn_expected_se

print("Spread of predictions (RMS of atoms, in A):")
print("   actual patch: {:1.2f}   TP: {:1.2f}   FN: {:1.2f}".format(p_spread, tp_spread, fn_spread))
print("   TP_boot_avg: {:1.2f}  ({:1.2f});   actual TP zscore (sd's): {:1.2f}".format(tp_expected_avg, tp_expected_se, z_tp))
print("   FN_boot_avg: {:1.2f}  ({:1.2f});   actual FN zscore (sd's): {:1.2f}".format(fn_expected_avg, fn_expected_se, z_fn))


