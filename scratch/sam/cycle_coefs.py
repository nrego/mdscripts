from __future__ import division

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
import sympy

import itertools

from scratch.sam.util import *

# Get rank of feature's associated cov matrix (max # linearly independent features)
def get_rank(feat):
    d = feat - feat.mean(axis=0)
    cov = np.dot(d.T, d)

    return np.linalg.matrix_rank(cov)

def gaus_elim(mat, indices_to_keep, indices_to_remove):
    root_indices = np.array([0,1,2])

    tmp_mat = mat.copy()

    B = np.zeros((3,3))

    for idx in indices_to_remove[:3]:
        this_col = tmp_mat[:,idx]

        # Now go row by row, starting from top and adding rows below as necessary
        for i_row in range(3):

            if i_row in indices_to_keep:
                continue

            if this_col[i_row] == 0:
                continue

            # One of original coeficients - don't remove
            if idx in root_indices and i_row == idx:
                continue

            else:
                print("doing")

            this_row = tmp_mat[i_row]

            

## Cycle through sets of linearly independent coefficients and recast the model ####
####################################################################################

# Set up our underdetermined system of linear constraints: Ax = b
## Our 'x' vector of variables is:
#     x = ( k_o, n_oo, n_oe, k_c, n_cc, n_ce, n_oc )

pq, p, q, k_o, n_oo, n_oe, k_c, n_cc, n_ce, n_oc = sympy.symbols('pq p q k_o n_oo n_oe k_c n_cc n_ce n_oc')
# ko, noo, noe
base_indices = np.array([0,1,2,3,4,5])

#x = np.array([pq, p, q, k_o, n_oo, n_oe, k_c, n_cc, n_ce, n_oc])
x = np.array([k_o, n_oo, n_oe, k_c, n_cc, n_ce, n_oc])

C = np.array([[6,-2,-1, 0, 0, 0,-1],
              [1, 0, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 1, 0],
              [0, 0, 0, 6,-2,-1,-1]], dtype=float)

## Get it in reduced row-echelon form
C[1] = 6*C[1] - C[0] + 2*C[3]
C[2] = C[2] + 2*C[3]
C[0] = C[0] + C[3]

#A = np.hstack((np.zeros((4,3)), A))
#b = np.array([pq, 0, 4*(p+q)-2, 0])
b = np.array([0,0,0,0])

lhs = np.dot(C, x.reshape(-1,1))

#equations = [sympy.Eq(left.item(), right) for left, right in zip(lhs, b)]
#res = sympy.solve(equations)


# inter in PQ, P, Q
reg_inter = np.load('sam_reg_inter.npy').item()
# Coef in k_o, n_oo, n_oe
reg_coef = np.load('sam_reg_coef.npy').item()


## Grab all data sets, for sanity
ds_pattern = np.load('sam_pattern_pooled.npz')
energies = ds_pattern['energies']
dg_bind = np.zeros_like(energies)
feat_vec = ds_pattern['feat_vec']

# Make feat_agree with our variables, above (append PQ to left)
feat_vec = np.hstack((feat_vec[:,:2].prod(axis=1).reshape(-1,1), feat_vec))


ds_bulk = np.load('sam_pattern_bulk_pure.npz')
bulk_p = ds_bulk['pq'][:,0]
bulk_q = ds_bulk['pq'][:,1]
## Fill up binding free energy
for i, energy in enumerate(energies):
    this_feat = feat_vec[i]
    this_p, this_q = this_feat[[1,2]]

    idx = np.where((bulk_p==this_p)&(bulk_q==this_q))[0].item()
    dg_bulk = ds_bulk['energies'][idx]
    dg_bind[i] = energy - dg_bulk


reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(feat_vec[:,base_indices], dg_bind)
reg.coef_[:3] = reg_inter.coef_
reg.coef_[3:] = reg_coef.coef_

pred = reg.predict(feat_vec[:,base_indices])
err = dg_bind - pred


indices_to_choose = np.arange(7)
# Reduced feature vector, remove PQ, P, Q, and non 6x6 patterns
feat_red = feat_vec[:884,3:]
## Should be 3
max_rank = get_rank(feat_red)
assert max_rank == 3
alpha = np.zeros(feat_red.shape[1])
alpha[:3] = reg_coef.coef_


groups = []

for idx in itertools.combinations(indices_to_choose, max_rank):
    idx = np.array(idx)

    this_feat = feat_red[:,idx]
    this_rank = get_rank(this_feat)

    if this_rank < max_rank:
        continue

    #print(x[idx])
    groups.append(x[idx])


b_mat = np.zeros((len(groups), 3, 3))
v = np.zeros((len(groups), 3), dtype=object)
ne = 4*(p+q) - 2
n = pq

c = (1/6)

for i in range(8,len(groups)):
    grp = groups[i]
    print('\n({}), {}'.format(i+1, grp))
    print('####################\n')

    this_b_mat =  np.array(eval(input("  Input matrix:\n")))
    this_v = np.array(eval(input("  Input v vals:\n")))

    b_mat[i,...] = this_b_mat
    v[i] = this_v
