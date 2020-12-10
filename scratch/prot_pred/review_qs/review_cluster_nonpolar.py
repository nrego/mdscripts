from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc
from matplotlib import pyplot as plt 
import numpy as np
import os, glob

from scipy.spatial import cKDTree

import sklearn.cluster

import time

### PLOT NAIVE NONPOLAR CLUSTERING ROCS, compare with reg##

## RUN FROM [prot]/old_prot_all/bound Directory ##
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 35})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':25})

from constants import k

def get_indices(beta_phi_old, beta_phi):
    indices = np.zeros_like(beta_phi_old).astype(int)

    for i, bphi in enumerate(beta_phi_old):
        try:
            idx = np.argwhere(beta_phi == bphi).item()
        except ValueError:
            idx = beta_phi.size - 1
        indices[i] = idx

    return indices


def harmonic_avg(tpr, tnr):
    recip_avg = ((1/tpr) + (1/tnr)) / 2

    return 1/recip_avg

def get_rg(positions):

    centroid = positions.mean(axis=0)
    diff = positions - centroid
    diff_sq = diff**2
    rsq = diff_sq.sum(axis=1)

    rmse = np.sqrt(rsq.mean())

    return rmse

# Add atom to existing cluster, given by mask, that results in smallest rg newcluster
def add_group(non_polar_atoms, this_mask):
    #current non-polar cluster
    current_grp = non_polar_atoms[this_mask]
    current_rg = get_rg(current_grp.positions)

    min_rg = np.inf
    for i in range(non_polar_atoms.n_atoms):

        # Non-polar atom i already in this cluster
        if this_mask[i]:
            continue

        test_mask = this_mask.copy()
        test_mask[i] = True

        test_grp = non_polar_atoms[test_mask]
        test_rg = get_rg(test_grp.positions)

        if test_rg < min_rg:
            min_rg = test_rg
            min_mask = test_mask

    return min_mask

min_clust_size = 7

homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':20})

from constants import k

buried_mask = np.loadtxt("buried_mask.dat", dtype=bool)
surf_mask = ~buried_mask 
non_polar_mask = np.loadtxt("hydropathy_mask.dat", dtype=bool)
contact_mask = np.loadtxt("actual_contact_mask.dat", dtype=bool)

# From local index of surface np atoms to global index of all heavy atoms
np_lut = np.arange(contact_mask.size)[non_polar_mask]

# Positives...
non_polar_contact_mask = contact_mask & non_polar_mask

univ = MDAnalysis.Universe("hydropathy.pdb")
prot = univ.atoms
prot.tempfactors = -2

non_polar_atoms = prot[non_polar_mask]
n_clust = non_polar_atoms.n_atoms // min_clust_size

tree = cKDTree(non_polar_atoms.positions)

dmat = tree.sparse_distance_matrix(tree, max_distance=100.0).toarray()

for i_round in range(1):
    print("CLUSTER ROUND: {}".format(i_round))    
    print("starting clustering...")
    start_time = time.time()
    clust = sklearn.cluster.SpectralClustering(n_clusters=n_clust).fit(non_polar_atoms.positions)
    end_time = time.time()
    print("...done ({:.2f} s)".format(end_time-start_time))

    min_rmse = np.inf
    min_mask = None

    for i in range(n_clust):
        mask = clust.labels_ == i

        if mask.sum() < min_clust_size:
            continue

        grp = non_polar_atoms[mask]
        rmse = get_rg(grp.positions)

        if rmse < min_rmse:
            min_rmse = rmse
            min_mask = mask.copy()
            min_i = i

        print("Clust: i={}. size: {} Rg: {:.2f}".format(i, grp.n_atoms, rmse))


    print("\nROUND: {}. min_rmse: {:.2f}".format(i_round, min_rmse))
    min_grp = non_polar_atoms[min_mask]
    min_grp.tempfactors = 1
    prot.write("clust_nucleus_{}.pdb".format(i_round))

min_size = min_grp.n_atoms

masks = np.zeros((non_polar_atoms.n_atoms - min_size + 1, non_polar_atoms.n_atoms), dtype=bool)
tp = np.zeros(non_polar_atoms.n_atoms - min_size + 1)
fp = np.zeros_like(tp)

masks[0] = min_mask

this_pred_contact_mask = np.zeros(prot.n_atoms, dtype=bool)

global_indices = np_lut[min_mask]
this_pred_contact_mask[global_indices] = 1

tp[0] = (this_pred_contact_mask & non_polar_contact_mask).sum()
fp[0] = (this_pred_contact_mask & ~non_polar_contact_mask).sum()


for i_round in range(1, non_polar_atoms.n_atoms - min_size + 1):
    min_mask = add_group(non_polar_atoms, min_mask)

    masks[i_round] = min_mask

    # Shape: n_heavies
    this_pred_contact_mask = np.zeros(prot.n_atoms, dtype=bool)

    global_indices = np_lut[min_mask]
    this_pred_contact_mask[global_indices] = 1

    tp[i_round] = (this_pred_contact_mask & non_polar_contact_mask).sum()
    fp[i_round] = (this_pred_contact_mask & ~non_polar_contact_mask).sum()


n_non_polar_contacts = non_polar_contact_mask[surf_mask].sum()
n_non_polar_non_contacts = (non_polar_mask[surf_mask] * ~contact_mask[surf_mask]).sum()

n_tot_contacts = contact_mask[surf_mask].sum()
n_tot_non_contacts = (~contact_mask[surf_mask]).sum()

tpr_np = tp / n_non_polar_contacts
fpr_np = fp / n_non_polar_non_contacts

norm_auc_np = np.trapz(tpr_np, fpr_np) / (tpr_np[-1] * fpr_np[-1])

tpr_all = tp / n_tot_contacts
fpr_all = fp / n_tot_non_contacts

norm_auc_all = np.trapz(tpr_all, fpr_all) / (tpr_all[-1] * fpr_all[-1])

d_h_np = harmonic_avg(tpr_np, 1-fpr_np)
d_h_all = harmonic_avg(tpr_all, 1-fpr_np)
max_idx_np = np.argmax(d_h_np)
max_idx_all = np.argmax(d_h_all)

### Now Load in regular ROC ##
beta_phi_old, tp_other, fp_other, tn_other, fn_other, tpr_other_all, fpr_other_all, prec, d_h_other_all, f_1, mcc = [arr.squeeze() for arr in np.split(np.loadtxt('../pred_reweight/performance.dat'), 11, 1)]
beta_phi, tp_other_np, tp_other_po, fp_other_np, fp_other_po, tn_other_np, tn_other_po, fn_other_np, fn_other_po = [arr.squeeze() for arr in np.split(np.loadtxt('../pred_reweight/perf_by_chemistry.dat'), 9, 1)]

indices = get_indices(beta_phi_old, beta_phi)

tpr_other_np = tp_other_np[indices] / n_non_polar_contacts
fpr_other_np = fp_other_np[indices] / n_non_polar_non_contacts

d_h_other_np = harmonic_avg(tpr_other_np, 1-fpr_other_np)

max_idx_other_np = np.argmax(d_h_other_np)
max_idx_other_all = np.argmax(d_h_other_all)

plt.close('all')
fig, ax = plt.subplots(figsize=(8,8))

ax.plot(fpr_other_all, tpr_other_all, 'o-', color=colors[0], label=r'$\phi$-ens, all contacts')
ax.plot(fpr_other_all[max_idx_other_all], tpr_other_all[max_idx_other_all], 'X', color=colors[0], markersize=20)

ax.plot(fpr_all, tpr_all, 'k--', linewidth=2, label=r'cluster, all contacts')
ax.plot(fpr_all[max_idx_all], tpr_all[max_idx_all], 'X', color=colors[0], markersize=20)

ax.plot(fpr_other_np, tpr_other_np, 'o-', color=colors[1], label=r'$\phi$-ens, non-polar contacts')
ax.plot(fpr_other_np[max_idx_other_np], tpr_other_np[max_idx_other_np], 'X', color=colors[1], markersize=20)

ax.plot(fpr_np, tpr_np, '--', color=colors[1], label=r'cluster, non-polar contacts')
ax.plot(fpr_np[max_idx_np], tpr_np[max_idx_np], 'X', color=colors[1], markersize=20)

ax.set_xlim(-0.02,1)
ax.set_ylim(0,1)
ax.set_xticks([0,0.5,1])
ax.set_yticks([0,0.5,1])

ax.set_xticklabels([])
ax.set_yticklabels([])


fig.tight_layout()
#plt.legend(loc=4)

plt.savefig('{}/Desktop/roc.pdf'.format(homedir))

norm_auc_other_all = np.trapz(tpr_other_all, fpr_other_all) / (tpr_other_all[-1] * fpr_other_all[-1])
norm_auc_other_np = np.trapz(tpr_other_np, fpr_other_np) / (tpr_other_np[-1] * fpr_other_np[-1])

plt.close('all')

print("\ndh_max_all: {:.2f}".format(d_h_all.max()))
print("AUC all: {:.2f}".format(norm_auc_all))

print("\ndh_max_other_all: {:.2f}".format(d_h_other_all.max()))
print("AUC other all: {:.2f}".format(norm_auc_other_all))

print("\ndh_max_np: {:.2f}".format(d_h_np.max()))
print("AUC np: {:.2f}".format(norm_auc_np))

print("\ndh_max_other_np: {:.2f}".format(d_h_other_np.max()))
print("AUC other np: {:.2f}".format(norm_auc_other_np))



## Save clustering roc perf
dat = np.vstack((tpr_all, fpr_all, d_h_all, tpr_np, fpr_np, d_h_np)).T 
np.savetxt("../pred_reweight/clust_perf.dat", dat)

dat = np.vstack((tpr_other_np, fpr_other_np, d_h_other_np))
np.savetxt("../pred_reweight/perf_np.dat", dat)


