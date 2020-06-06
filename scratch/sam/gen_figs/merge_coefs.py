
import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

import itertools

from sklearn import datasets, linear_model
from sklearn.cluster import AgglomerativeClustering

## Merge edge classes (h_k^oo for each edge k) into 'classes' of similar edges
#   Using a greedy algorithm (Best R2 performance after merging)
#   
#   Initially, all external edges made by each peripheral node are merged (since they're equiv anyway)
#
#

def bootstrap(dataset, fn, n_boot=1000):
    np.random.seed()

    assert dataset.ndim == 1
    n_dat = dataset.size

    boot_samples = np.zeros((n_boot))

    for i_boot in range(n_boot):
        this_boot = np.random.choice(dataset, n_dat)

        boot_samples[i_boot] = fn(this_boot)


    return boot_samples

def unpackbits(x, num_bits):
  xshape = list(x.shape)
  x = x.reshape([-1, 1])
  to_and = 2**np.arange(num_bits).reshape([1, num_bits])

  return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

# Construct all possible states with given size
def build_states(p=2, q=2):
    n = p*q
    n_states = 2**n

    #if n > 8:
    #    print("too many states, exiting")

    states = np.empty(n_states, dtype=object)

    indices = np.arange(n)

    methyl_masks = unpackbits(np.arange(n_states), n).astype(bool)

    for i, methyl_mask in enumerate(methyl_masks):

        states[i] = State(indices[methyl_mask], p, q)

    return states

def get_rank(feat):
    d = feat - feat.mean(axis=0)
    cov = np.dot(d.T, d)

    return np.linalg.matrix_rank(cov)

# Construct a reduced feature set by summing over features
#    with like labels
def construct_red_feat(feat_vec, labels):

    n_sample = feat_vec.shape[0]
    # Number of unique features after merging based on labels
    n_clust = np.unique(labels).size

    red_feat = np.zeros((n_sample, n_clust))

    for i_clust in range(n_clust):
        mask = (labels == i_clust)
        red_feat[:, i_clust] = feat_vec[:,mask].sum(axis=1)

    return red_feat

## Get a full-edge specified feature vector
#    (over-parameterized)
#    shape: (3*M_tot, )
#
def get_full_feat_vec(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    
    idx = 3*states[0].n_edges
    n_feat = idx
    feat_vec = np.zeros((n_sample, n_feat))

    for i,state in enumerate(states):

        feat_vec[i, 0:idx:3] = state.edge_oo
        feat_vec[i, 1:idx:3] = state.edge_oc
        feat_vec[i, 2:idx:3] = state.edge_cc
        #feat_vec[i, -1] = state.k_o


    return feat_vec

# Shape: (M_int+1,)
#   hoo_k of each edge (all M_int of them), plus ko
def get_feat_vec_oo(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    # ko, nkoo, and nkcc, so a total of 2*M_tot + 1 coef's
    
    n_feat = n_edge+1
    feat_vec = np.zeros((n_sample, n_feat))

    for i,state in enumerate(states):

        feat_vec[i, 0:n_edge] = state.edge_oo
        feat_vec[i, -1] = state.k_o


    return feat_vec

# Shape (M_int+N_ext), hk_oo for each internal edge, followed by 
#   hydroxyl mask for each peripheral node

def test_get_feat_vec(states):
    n_sample = states.size
    n_edge = states[0].n_edges
    n_feat = states[0].M_int + states[0].N_tot

    feat_vec = np.zeros((n_sample, n_feat))

    for i,state in enumerate(states):
        hydroxyl_mask = ~state.methyl_mask

        feat_vec[i,:state.M_int] = state.edge_oo[state.edges_int_indices]
        feat_vec[i,state.M_int:] = hydroxyl_mask

    return feat_vec

# Shape: (M_int+N_ext+1)
#    h_koo for each internal edge, followed by number of external oo edges for
#.     each periph node (i.e., h_iooext times miext for each periph node i)
#.     followed by number of total hydroxyls
def test_get_feat_vec2(states):
    n_sample = states.size
    n_feat = states[0].M_int + states[0].N_ext + 1

    idx1 = states[0].M_int
    idx2 = idx1 + states[0].N_ext

    feat_vec = np.zeros((n_sample, n_feat))

    for i, state in enumerate(states):
        hydroxyl_mask = ~state.methyl_mask

        feat_vec[i, :idx1] = state.edge_oo[state.edges_int_indices]
        feat_vec[i, idx1:idx2] = state.ext_count[state.ext_indices] * hydroxyl_mask[state.ext_indices]
        feat_vec[i, idx2:] = state.k_o

    return feat_vec





def aic(n_samples, mse, k_params, do_corr=False):

    if do_corr:
        corr = (2*k_params**2 + 2*k_params) / (n_samples-k_params-1)
    else:
        corr = 0

    return n_samples*np.log(mse) + 2*k_params + corr

def plot_1d(coef, labels=None):
    if labels is None:
        labels = np.ones_like(coef)
    plt.close()
    plt.scatter(coef, np.ones_like(coef), c=labels)

# For each edge k, find its equivalent edge (thru symmetry) l
def find_sym_edges(state):
    p, q = state.p, state.q
    # Lut of symmetric nodes.
    #    node (local) patch index i gives (local) patch index j, its equivalent node
    #    This numbering is dependent on whether p is odd or even
    rev_node_lut = np.zeros(state.N_tot, dtype=int)

    # Even p, this is the easy case - can just reverse numbering
    if p % 2 == 0:
        rev_node_lut = state.N_tot - np.arange(state.N_tot) - 1

    # odd p; vertical sym axis down middle. 
    else:
        indices = np.arange(state.N_tot)
        for i_col in range(p):
            # index of equivalent column
            rev_col_idx = p-1-i_col
            rev_node_lut[i_col*q:(i_col+1)*q] = indices[rev_col_idx*q:(rev_col_idx+1)*q]

    # Edge lookup, indexed by edge k, gives sym edge l
    rev_edge_lut = np.zeros(state.n_edges, dtype=int)

    # Memoizing each external edge we visit for each (local) i
    n_ext_indices = np.zeros(state.N_tot, dtype=int)

    for idx_k, (i,j) in enumerate(state.edges):
        local_i = np.argwhere(i==state.patch_indices)[0].item()

        # (local) index of i's symmetric node
        rev_local_i = rev_node_lut[local_i]
        rev_global_i = state.patch_indices[rev_local_i]

        j_int = j in state.patch_indices

        # internal edge
        if j_int:
            local_j = np.argwhere(j==state.patch_indices)[0].item()
            
            # (local) index of j's sym node
            rev_local_j = rev_node_lut[local_j]
            rev_global_j = state.patch_indices[rev_local_j]

            mask = ((state.edges[:,0] == rev_global_i) & (state.edges[:,1] == rev_global_j)) | ((state.edges[:,1] == rev_global_i) & (state.edges[:,0] == rev_global_j)) 
            assert mask.sum() == 1

            idx_l = np.arange(state.n_edges)[mask].item()

            rev_edge_lut[idx_k] = idx_l

        # External edge
        else:
            this_ext_idx = n_ext_indices[local_i]

            # Who knows why reverse edge indexing has to be reversed, but it really doesnt matter
            #.   since all external edges formed by node i are equivalent, anyway
            #if p % 2 == 0:
            idx_l = state.nodes_to_ext_edges[rev_local_i][::-1][this_ext_idx]
            #else:
                #idx_l = state.nodes_to_ext_edges[rev_local_i][this_ext_idx]

            rev_edge_lut[idx_k] = idx_l
            n_ext_indices[local_i] += 1

    assert np.array_equal(state.ext_count, n_ext_indices)

    labels = np.zeros(state.M_int+state.N_ext, dtype=int)
    labels[:] = -1

    i_label = 0
    # For each internal edge (indexed k)
    for local_k, k_edge_internal in enumerate(state.edges_int_indices):
        
        assert local_k == np.argwhere(state.edges_int_indices==k_edge_internal).item() 

        # Already dealt with this edge (thru symmetry)
        if labels[local_k] > -1:
            continue

        # Symmetrically equivalent edge l
        l_edge_internal = rev_edge_lut[k_edge_internal]

        local_l = np.argwhere(state.edges_int_indices==l_edge_internal).item()

        labels[local_k] = i_label
        labels[local_l] = i_label

        i_label += 1

    assert labels[:state.M_int].min() == 0
    assert labels[:state.M_int].max() == state.M_int // 2

    # Now peripheral nodes
    for i_local_ext, i_ext_idx in enumerate(state.ext_indices):

        this_idx = state.M_int+i_local_ext

        if labels[this_idx] > -1:
            continue
        
        # (local) patch index j of equivalent node to i
        j_ext_idx = rev_node_lut[i_ext_idx]

        j_local_ext = np.argwhere(state.ext_indices==j_ext_idx).item()

        labels[this_idx] = i_label
        labels[state.M_int+j_local_ext] = i_label

        i_label += 1


    return rev_node_lut, rev_edge_lut, labels


# When merging from mgc0 to mgc1, find 
#.   which groups are merged, what their coefs were,
#.   and what the new coef is
def check_merged_coef(mgc0, mgc1, feat_vec, energies):
    
    # find indices of merged groups
    for t_idx1 in range(len(mgc0)):
        for t_idx2 in range(t_idx1+1, len(mgc0)):
            for t_idx_merge in range(len(mgc1)):
                if np.array_equal(np.sort(np.append(mgc0.groups[t_idx1].indices, mgc0.groups[t_idx2].indices)), np.sort(mgc1.groups[t_idx_merge].indices)):
                    idx1 = t_idx1
                    idx2 = t_idx2
                    idx_merge = t_idx_merge
                    break

    assert np.array_equal(np.sort(np.append(mgc0.groups[idx1].indices, mgc0.groups[idx2].indices)), np.sort(mgc1.groups[idx_merge].indices))

    labels0 = mgc0.labels
    labels1 = mgc1.labels

    reg0 = linear_model.LinearRegression()
    reg1 = linear_model.LinearRegression()

    tmp_feat0 = construct_red_feat(feat_vec, np.append(labels0, labels0.max()+1))
    tmp_feat1 = construct_red_feat(feat_vec, np.append(labels1, labels1.max()+1))

    reg0.fit(tmp_feat0, energies)
    reg1.fit(tmp_feat1, energies)

    print("merging groups {} and {} to {}".format(idx1, idx2, idx_merge))
    print("Merge groups {} and {}".format(mgc0.groups[idx1], mgc0.groups[idx2]))
    print("Coefs: {:.2f} and {:.2f} -> {:.2f}".format(reg0.coef_[idx1], reg0.coef_[idx2], reg1.coef_[idx_merge]))


    return (reg0, reg1)



# Partition data into k groups
#def partition_feat(feat_vec, k=3)

### Merge edge types
k_cv=5

p = 6
q = 6

fname = 'data/sam_pattern_{:02d}_{:02d}.npz'.format(p,q)

energies, ols_feat_vec, states = extract_from_ds(fname)
err_energies = np.load(fname)['err_energies']
n_dat = energies.size

# grab a pure methyl state
state = states[np.argwhere(ols_feat_vec[:,0] == 0).item()]

sym_node_lut, sym_edge_lut, sym_labels = find_sym_edges(state)

aug_states = np.empty(2*n_dat, dtype=object)
aug_energies = np.zeros(2*n_dat)

for i, tmp_state in enumerate(states):
    aug_states[i] = tmp_state
    inv_state = State(sym_node_lut[tmp_state.pt_idx], p=tmp_state.p, q=tmp_state.q)

    aug_states[i+n_dat] = inv_state

    aug_energies[i] = energies[i]
    aug_energies[i+n_dat] = energies[i]

#states = aug_states.copy()
#energies = aug_energies.copy()
#ols_feat_vec = np.append(ols_feat_vec, ols_feat_vec, axis=0)

## Over parameterized, shape: (3*M_tot, )
#  h_oo, h_cc, h_oo for each edge location
feat_vec1 = get_full_feat_vec(states)

# Shape: M_tot+1 (hkoo for each edge, plus ko)
feat_vec2 = get_feat_vec_oo(states)

# shape: M_int+N_ext+1 (hkoo for each internal, number of external oo edges for each periph, plus ko)
feat_vec3 = test_get_feat_vec2(states)




# Full feature vector (edge type for each edge, 3*M_tot)
perf1, err1, _, _, reg1 = fit_k_fold(feat_vec1, energies, k=k_cv, do_ridge=True)
# h_oo (one for each of M_tot edges), plus k_o. Has redundancies (periph edges)
perf2, err2, _, _, reg2 = fit_k_fold(feat_vec2, energies, k=k_cv, do_ridge=True)
# ko, sum over all internal edges, sum over all external edges (miext edges for each peripheral node); shape: (M_int + N_periph + 1)
perf3, err3, _, _, reg3 = fit_k_fold(feat_vec3, energies, k=k_cv)
perf_m3, err_m3, _, _, reg_m3 = fit_k_fold(ols_feat_vec, energies, k=k_cv) 

perf_m1, err_m1, _, _, reg_m1 = fit_k_fold(ols_feat_vec[:,0].reshape(-1,1), energies, k=k_cv)

### Merge and label classes of edges ###
########################################

n_clust = 1
mgc_0 = MergeGroupCollection()

# Each internal edge gets own group...
[ mgc_0.add_group(MergeGroup(k, label='internal')) for k in state.edges_int_indices ]

for ext_k in state.nodes_to_ext_edges[state.ext_indices]:
    mgc_0.add_group(MergeGroup(ext_k, label='external'))

mgc = copy.deepcopy(mgc_0)


all_mgc = []
all_mse = []
all_n_params = []

reg = linear_model.LinearRegression()

min_mgc = copy.deepcopy(mgc)
temp_feat_vec = construct_red_feat(feat_vec2, np.append(min_mgc.labels, min_mgc.labels.max()+1))

reg.fit(temp_feat_vec, energies)
min_mse = np.mean((energies - reg.predict(temp_feat_vec))**2)

i_merge = 0

## Now perform merging until we have the requisite number of groups
while len(mgc) >= n_clust:
    print("merge round i:{}".format(i_merge))
    
    all_mgc.append(min_mgc)
    all_mse.append(min_mse)
    # Number of edge classes, plus ko, plus intercept
    all_n_params.append(len(min_mgc)+2)

    min_mse = np.inf
    min_mgc = None
    min_i = -1
    min_j = -1

    for i_grp, j_grp in itertools.combinations(np.arange(len(mgc)), 2):

        # Only two groups remain - hack so they can be merged
        if len(mgc) == 2:
            mgc.groups[i_grp].label = mgc.groups[j_grp].label = 'edge'

        if mgc.groups[i_grp].label != mgc.groups[j_grp].label:
            continue

        # Try this merge, test r2
        temp_mgc = copy.deepcopy(mgc)

        temp_mgc.merge_groups(i_grp, j_grp)

        temp_labels = temp_mgc.labels
        temp_feat_vec = construct_red_feat(feat_vec2, np.append(temp_labels, temp_labels.max()+1))

        reg.fit(temp_feat_vec, energies)
        temp_mse = np.mean((energies - reg.predict(temp_feat_vec))**2)

        # Best merge so far this round
        if temp_mse < min_mse:
            min_mse = temp_mse
            min_mgc = temp_mgc
            min_i = i_grp
            min_j = j_grp


        del temp_feat_vec, temp_mgc

    ## Now merge 
    if min_i == min_j == -1:
        break
    mgc.merge_groups(min_i, min_j)
    i_merge += 1


all_mse = np.array(all_mse)
all_n_params = np.array(all_n_params)
all_cv_mse = np.zeros_like(all_mse)

# Do the CV here, as a final check
for i, this_mgc in enumerate(all_mgc):
    this_labels = np.append(this_mgc.labels, this_mgc.labels.max()+1)

    red_feat = construct_red_feat(feat_vec2, this_labels)
    perf, err, _, _, this_reg = fit_k_fold(red_feat, energies, k=k_cv)

    all_cv_mse[i] = perf.mean()


myaic = aic(n_dat, all_mse, all_n_params, do_corr=True)

np.savez_compressed('merge_data/sam_merge_coef_class_{:02d}_{:02d}'.format(p,q), all_mse=all_mse, 
                    all_n_params=all_n_params, all_cv_mse=all_cv_mse, all_mgc=all_mgc, feat_vec=feat_vec2)

cmap = plt.cm.tab20
