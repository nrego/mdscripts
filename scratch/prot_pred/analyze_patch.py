from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from IPython import embed
from mdtools import MDSystem
import cPickle as pickle
import argparse

import os

from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

homedir = os.environ['HOME']


mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':20})

COLOR_CONTACT = '#7F00FF' # purple
COLOR_NO_CONTACT = COLOR_NOT_PRED = COLOR_TN = '#7F7F7F' # gray
COLOR_PHIL = '#0560AD' # blue2
COLOR_PHOB = '#D7D7D7' # light gray
COLOR_PRED = '#FF7F00' # orange 
COLOR_TP = '#FF007F' # pink
COLOR_FP = '#7F3F00' # dark orange
COLOR_FN = '#3F007F' # dark purple

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


def make_piechart(slices, colors, ecolors, outname, showtext=True, groups=[[0,1],[2,3]], radfraction=0.2):
    assert len(groups) == 2
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5,5))
    wedgeprops = {'edgecolor':'k', 'linewidth':6}

    patches, texts, pcts = ax.pie(slices, labeldistance=1.15, colors=colors, autopct=lambda p: '{:1.1f}\%'.format(p), wedgeprops=wedgeprops)
    [pct.set_fontsize(20) for pct in pcts]
    ax.axis('equal')

    for patch, ecolor in zip(patches, ecolors):
        patch.set_edgecolor(ecolor)

    fig.savefig('{}'.format(outname), transparent=True)
    plt.close('all')


    fig, ax = plt.subplots(figsize=(5,5))
    ax.axis('equal')
    ax.axis('off')
    #embed()
    i = groups[0]
    ang = np.deg2rad((patches[i[-1]].theta2 + patches[i[0]].theta1)/2.0)
    wedges = []
    for j in i:
        patch = patches[j]
        center = (radfraction*patch.r*np.cos(ang), radfraction*patch.r*np.sin(ang))
        wedges.append(mpatches.Wedge(center, patch.r, patch.theta1, patch.theta2, edgecolor=patch.get_edgecolor(), facecolor=patch.get_facecolor()))
        text_ang = np.deg2rad((patch.theta1 + patch.theta2)/2.0)
        if text_ang > np.pi and text_ang < (3*np.pi)/2.0:
            align='top'
            text_pos = np.array(((1.12)*patch.r*np.cos(text_ang), (1.12)*patch.r*np.sin(text_ang))) + np.array(center)
        else:
            align='baseline'
            text_pos = np.array(((1.08)*patch.r*np.cos(text_ang), (1.08)*patch.r*np.sin(text_ang))) + np.array(center)
        if showtext:
            ax.text(*text_pos, s=pcts[j].get_text(), fontsize=20, verticalalignment=align)
    i = groups[1]
    for j in i:
        try:
            patch = patches[j]
        except:
            embed()
        center = patch.center
        
        wedges.append(mpatches.Wedge(center, patch.r, patch.theta1, patch.theta2, edgecolor=patch.get_edgecolor(), facecolor=patch.get_facecolor()))
        text_ang = np.deg2rad((patch.theta1 + patch.theta2)/2.0)
        if text_ang > np.pi and text_ang < (3*np.pi)/2.0:
            align='top'
            text_pos = np.array(((1.12)*patch.r*np.cos(text_ang), (1.12)*patch.r*np.sin(text_ang))) + np.array(center)
        else:
            align='baseline'
            text_pos = np.array(((1.08)*patch.r*np.cos(text_ang), (1.08)*patch.r*np.sin(text_ang))) + np.array(center)
        if showtext:
            ax.text(*text_pos, s=pcts[j].get_text(), fontsize=20, verticalalignment=align)

    collection = PatchCollection(wedges)
    collection.set_facecolors(colors)
    collection.set_linewidths(6)
    collection.set_edgecolors(ecolors)

    #collection.set_array(np.array(colors))
    ax.add_collection(collection)
    ax.autoscale(True)
    #ax.text(*pcts[0].get_position(), s=pcts[0].get_text())

    fig.savefig('{}_nolabel'.format(outname), transparent=False)

    plt.close('all')

## Analyze composition of classified predictions

parser = argparse.ArgumentParser('analyze atomic composition of TPs, FPs, etc')
parser.add_argument('-s', '--top', type=str, required=True,
                    help='Topology (TPR) file')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Structure (GRO) file')
parser.add_argument('--sel-spec', type=str, default='segid seg_0_Protein_targ and not name H*',
                    help='Selection spec to get target heavy atoms')
parser.add_argument('--actual-contact', type=str, default='../../bound/actual_contact_mask.dat',
                    help='Actual contacts (Default: %(default)s)')
parser.add_argument('--buried', type=str, default='../../bound/buried_mask.dat',
                    help='Buried mask (Default: %(default)s)')
parser.add_argument('--pred-contact', type=str, default='pred_contact_mask.dat',
                    help='Predicted contact mask (Default: %(default)s)')
parser.add_argument('--hydropathy', type=str, default='../../bound/hydropathy_mask.dat',
                    help='Hydropathy mask (Default: %(default)s)')
parser.add_argument('--outpath', type=str, required=True,
                    help='Prefix for output')

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
n_surf = surf_mask.sum()
assert n_surf == targ.n_atoms

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

assert cond_pos == contact_mask.sum()

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

print("Fraction hydrophobic:")

surf_phob = hydropathy.sum() / n_surf
surf_phil = (n_surf - hydropathy.sum()) / n_surf

print("fraction of surface atoms that are hydrophobic: {:2.2f}".format(surf_phob))


##################################################
## Now make pie plot of actual contact composition
##################################################

# fraction of actual contacts that are hydrophobic/philic
contact_phob = (contact_mask & hydropathy).sum()/n_surf
contact_phil = (contact_mask & ~hydropathy).sum()/n_surf
no_contact_phob = (~contact_mask & hydropathy).sum()/n_surf
no_contact_phil = (~contact_mask & ~hydropathy).sum()/n_surf

print("fraction of actual contacts that are hydrophobic: {:2.2f}".format(contact_phob))
colors = (COLOR_CONTACT, COLOR_NO_CONTACT, COLOR_NO_CONTACT, COLOR_CONTACT)
ecolors = (COLOR_PHOB, COLOR_PHOB, COLOR_PHIL, COLOR_PHIL)
make_piechart([contact_phob, no_contact_phob, no_contact_phil, contact_phil], colors, ecolors, '{}_contact'.format(args.outpath))

##################################################
## Now make pie plot of predicted contact composition
##################################################

pred_phob = (pred_mask & hydropathy).sum()/n_surf
pred_phil = (pred_mask & ~hydropathy).sum()/n_surf
no_pred_phob = (~pred_mask & hydropathy).sum()/n_surf
no_pred_phil = (~pred_mask & ~hydropathy).sum()/n_surf

colors = (COLOR_PRED, COLOR_NOT_PRED, COLOR_NOT_PRED, COLOR_PRED)
ecolors = (COLOR_PHOB, COLOR_PHOB, COLOR_PHIL, COLOR_PHIL)
#make_piechart([pred_phob, no_pred_phob, no_pred_phil, pred_phil], colors, ecolors, '{}_pred'.format(args.outpath), showtext=False)
make_piechart([pred_phob, no_pred_phob, no_pred_phil, pred_phil], colors, ecolors, '{}_pred'.format(args.outpath))

##################################################
## Finally, of performance
##################################################

tp_phob = (tp_mask & hydropathy).sum()/n_surf
tp_phil = (tp_mask & ~hydropathy).sum()/n_surf
fp_phob = (fp_mask & hydropathy).sum()/n_surf
fp_phil = (fp_mask & ~hydropathy).sum()/n_surf
fn_phob = (fn_mask & hydropathy).sum()/n_surf
fn_phil = (fn_mask & ~hydropathy).sum()/n_surf
tn_phob = (tn_mask & hydropathy).sum()/n_surf
tn_phil = (tn_mask & ~hydropathy).sum()/n_surf

colors = (COLOR_TP, COLOR_FP, COLOR_FN, COLOR_TN, COLOR_TN, COLOR_FN, COLOR_FP, COLOR_TP)
ecolors = (COLOR_PHOB, COLOR_PHOB, COLOR_PHOB, COLOR_PHOB, COLOR_PHIL, COLOR_PHIL, COLOR_PHIL, COLOR_PHIL)
make_piechart([tp_phob, fp_phob, fn_phob, tn_phob, tn_phil, fn_phil, fp_phil, tp_phil], colors, ecolors, '{}_perf'.format(args.outpath), groups=[[0,1,2,3], [4,5,6,7]])



# Find fraction of groups that are phobic
#   Expectation: TP will be more hydrophobic than FNs (indicating that we are capturing the **hydrophobic** parts of patch)
tp_phob = (tp_mask & hydropathy).sum()/tp
fp_phob = (fp_mask & hydropathy).sum()/fp
tn_phob = (tn_mask & hydropathy).sum()/tn
fn_phob = (fn_mask & hydropathy).sum()/fn


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


