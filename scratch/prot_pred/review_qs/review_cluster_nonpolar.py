from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc
from matplotlib import pyplot as plt 
import numpy as np
import os, glob

from scipy.spatial import cKDTree

import sklearn.cluster

import time


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
mpl.rcParams.update({'legend.fontsize':30})

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
plt.close()
plt.scatter(fpr_np, tpr_np, label='non-polar')
plt.scatter(fpr_all, tpr_all, label='all')






