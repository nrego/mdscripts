from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc 
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from constants import k

from IPython import embed

import scipy

homedir = os.environ['HOME']
savedir = '{}/Desktop'.format(homedir)

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':15})

harmonic_mean = lambda x1, x2: (0.5*(x1**-1 + x2**-1))**-1

def plot_it(fname, bphi, dat1, dat2, label1, label2):
    #plt.close('all')
    fig, ax = plt.subplots(figsize=(6,5))

    ax.plot(bphi, dat1, label=label1, color=(0.5, 0.5, 0.5), linewidth=4)
    ax.plot(bphi, dat2, label=label2, color='#0060AA', linewidth=4)
    ax.set_xlim(0, 4.05)
    ax.set_ylim(0, 1.01)
    fig.legend()
    fig.tight_layout()
    fig.savefig('{}/{}'.format(savedir, fname), transparent=True)
    plt.close('all')    

beta_phi, tp_np, tp_po, fp_np, fp_po, tn_np, tn_po, fn_np, fn_po = np.split(np.loadtxt("perf_by_chemistry.dat"), 9, axis=1)

tp = tp_np + tp_po
fp = fp_np + fp_po
tn = tn_np + tn_po
fn = fn_np + fn_po

n_np = tp_np + fp_np + tn_np + fn_np
n_po = tp_po + fp_po + tn_po + fn_po

# P(np | dewetted) - fraction of dewetted atoms that are nonpolar
p_np_p = (tp_np + fp_np) / (tp + fp)
# P(po | dewetted) - fraction of dewetted atoms that are polar/charged
p_po_p = (tp_po + fp_po) / (tp + fp)
plot_it('p_dewet.png', beta_phi, p_np_p, p_po_p, r'$P(\mathrm{np} | +)$', r'$P(\mathrm{po} | +)$')

# P(dewetted | np) - fraction non-polar atoms that are dewetted
p_p_np = (tp_np + fp_np) / n_np
# P(dewetted | po) - fraction of polar atoms that are dewetted
p_p_po = (tp_po + fp_po) / n_po
plot_it('p_chem_dewet.png', beta_phi, p_p_np, p_p_po, r'$P(+|\mathrm{np})$', r'$P(+|\mathrm{po})$')

# P(np | ~dewetted) - fraction of wet atoms that are nonpolar
p_np_n = (tn_np + fn_np) / (tn + fn)
# P(po | ~dewetted) - given atom is wet, what's the probability it's polar/charged?
p_po_n = (tn_po + fn_po) / (tn + fn)
plot_it('p_not_dewet.png', beta_phi, p_np_n, p_po_n, r'$P(\mathrm{np} | -)$', r'$P(\mathrm{po} | -)$')

# P(~dewet | np) - fraction of non-polar atoms that are not dewetted (given an atom is np, what's chance it's wet?)
p_n_np = (tn_np + fn_np) / (n_np)
# P(~dewet | po) - given an atom is polar/charged, what's the probability it's wet?
p_n_po = (tn_po + fn_po) / n_po
plot_it('p_chem_not_dewet.png', beta_phi, p_n_np, p_n_po, r'$P(-|\mathrm{np})$', r'$P(-|\mathrm{po})$')

# Fraction of true positives that are non-polar or polar/charged
p_np_tp = (tp_np)/tp
p_po_tp = (tp_po)/tp
plot_it('p_chem_tp.png', beta_phi, p_np_tp, p_po_tp, r'$P(\mathrm{np} | \mathrm{TP})$', r'$P(\mathrm{po} | \mathrm{TP})$')

# Fraction of false positives that are non-polar or polar/charged
p_np_fp = (fp_np)/fp
p_po_fp = (fp_po)/fp
plot_it('p_chem_fp.png', beta_phi, p_np_fp, p_po_fp, r'$P(\mathrm{np} | \mathrm{FP})$', r'$P(\mathrm{po} | \mathrm{FP})$')


# Fraction of true negatives that are non-polar or polar/charged
p_np_tn = (tn_np)/tn
p_po_tn = (tn_po)/tn
plot_it('p_chem_tn.png', beta_phi, p_np_tn, p_po_tn, r'$P(\mathrm{np} | \mathrm{TN})$', r'$P(\mathrm{po} | \mathrm{TN})$')


# Fraction of false negatives that are non-polar or polar/charged
p_np_fn = (fn_np)/fn
p_po_fn = (fn_po)/fn
plot_it('p_chem_fn.png', beta_phi, p_np_fn, p_po_fn, r'$P(\mathrm{np} | \mathrm{FN})$', r'$P(\mathrm{po} | \mathrm{FN})$')


# TPR of non-polar and polar/charged atoms
##########################################
# I.e. probability that a non-polar contact atoms is a true positive
tpr_np = (tp_np) / (tp_np + fn_np)
tpr_po = (tp_po) / (tp_po + fn_po)
plot_it('tpr_chem.png', beta_phi, tpr_np, tpr_po, r'$\mathrm{TPR}\; (\mathrm{np})$', r'$\mathrm{TPR}\; (\mathrm{po})$')

# FPR of non-polar and polar/charged atoms
##########################################
fpr_np = (fp_np) / (fp_np + tn_np)
fpr_po = (fp_po) / (fp_po + tn_po)
plot_it('fpr_chem.png', beta_phi, fpr_np, fpr_po, r'$\mathrm{FPR}\; (\mathrm{np})$', r'$\mathrm{FPR}\; (\mathrm{po})$')


tpr_tot = (tp) / (tp + fn)
fpr_tot = (fp) / (fp + tn)
prec_tot = (tp) / (tp + fp)

fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(fpr_np, tpr_np, label='non-polar', color=(0.5,0.5,0.5))
ax.scatter(fpr_po, tpr_po, label='polar/charged', color='#0060AA')
ax.scatter(fpr_tot, tpr_tot, color='k', label='both')
fig.legend()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
fig.tight_layout()

fig.savefig('{}/roc_comparison.png'.format(savedir), transparent=True)
plt.close('all')

# PRECISION of non-polar and polar/charged atoms
################################################
# Fraction of non-polar dewetted atoms that are true positives
#    (given a dewetted atom is non-polar, what's the chance it's actually a contact?)
p_tp_np = (tp_np)/(tp_np + fp_np)
# Fraction of polar/charged dewetted atoms that are true positive
#    (given that we know a dewetted atom is polar/charged, what's the probability it's a contact?)
p_tp_po = (tp_po)/(tp_po + fp_po)

plot_it('prec_chem.png', beta_phi, p_tp_np, p_tp_po, r'$\mathrm{PPV}\; (\mathrm{np})$', r'$\mathrm{PPV}\; (\mathrm{po})$')


# Now plot TPR, FPR, and PPV for NP and PO atoms
plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(beta_phi, tpr_np, color=(0.5,0.5,0.5), linestyle='-', label='TPR', linewidth=4)
ax.plot(beta_phi, fpr_np, color=(0.5,0.5,0.5), linestyle='--', label='FPR', linewidth=4)
ax.plot(beta_phi, p_tp_np, color=(0.5,0.5,0.5), linestyle='-.', label='PPV', linewidth=4)

ax.plot(beta_phi, tpr_po, color='#0060AA', linestyle='-', label='TPR', linewidth=4)
ax.plot(beta_phi, fpr_po, color='#0060AA', linestyle='--', label='FPR', linewidth=4)
ax.plot(beta_phi, p_tp_po, color='#0060AA', linestyle='-.', label='PPV', linewidth=4)

fig.legend()
ax.set_xlim(1.6,4.05)
ax.set_ylim(0,1.01)
fig.tight_layout()
fig.savefig('{}/tpr_fpr_prec_comparison.png'.format(savedir))
plt.close('all')


f1_tot = harmonic_mean(tpr_tot, prec_tot)
dh_tot = harmonic_mean(tpr_tot, 1-fpr_tot)

f1_np = harmonic_mean(tpr_np, p_tp_np)
dh_np = harmonic_mean(tpr_np, 1-fpr_np)

f1_po = harmonic_mean(tpr_po, p_tp_po)
dh_po = harmonic_mean(tpr_po, p_tp_po)

plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(beta_phi, f1_np, color=(0.5,0.5,0.5), linewidth=4, label='$f_1\;\mathrm{np}$')
ax.plot(beta_phi, f1_po, color='#0060AA', linewidth=4, label='$f_1\;\mathrm{po}$')
ax.plot(beta_phi, f1_tot, color='k', linewidth=4, label='$f_1\;\mathrm{both}$')
fig.tight_layout()
ax.set_xlim(0,4.05)
ax.set_ylim(0,1)
fig.legend()
fig.savefig('{}/f1_comparison.png'.format(savedir), transparent=True)
plt.close('all')

plt.close('all')
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(beta_phi, dh_np, color=(0.5,0.5,0.5), linewidth=4, label='$d_h\;\mathrm{np}$')
ax.plot(beta_phi, dh_po, color='#0060AA', linewidth=4, label='$d_h\;\mathrm{po}$')
ax.plot(beta_phi, dh_tot, color='k', linewidth=4, label='$d_h\;\mathrm{both}$')
fig.tight_layout()
ax.set_xlim(0,4.05)
ax.set_ylim(0,1)
fig.legend()
fig.savefig('{}/dh_comparison.png'.format(savedir), transparent=True)
plt.close('all')


