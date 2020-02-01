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

            

## Cycle through sets of linearly independent coefficients and recast the model ####
####################################################################################

# Set up our underdetermined system of linear constraints: Ax = b
## Our 'x' vector of variables is:
#     x = ( k_o, n_oo, n_oe, k_c, n_cc, n_ce, n_oc )

ds = np.load('sam_bmat.npz')
vals_B = ds['B']
vals_v = ds['v']
groups = ds['groups']

pq, p, q, k_o, n_oo, n_oe, k_c, n_cc, n_ce, n_oc = sympy.symbols('pq p q k_o n_oo n_oe k_c n_cc n_ce n_oc')

x = np.array([k_o, n_oo, n_oe, k_c, n_cc, n_ce, n_oc])
tmp_x = np.array([pq, p, q, k_o, n_oo, n_oe, k_c, n_cc, n_ce, n_oc])

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
reg.fit(feat_vec[:,:6], dg_bind)
reg.coef_[:3] = reg_inter.coef_
reg.coef_[3:] = reg_coef.coef_

pred = reg.predict(feat_vec[:,:6])
err = dg_bind - pred


a1, a2, a3 = sympy.symbols('a1 a2 a3')
alpha = np.array([a1, a2, a3])
act_alpha = reg.coef_[3:]

f0 = reg.coef_[0]*36 + reg.coef_[1]*6 + reg.coef_[2]*6

for i, grp in enumerate(groups):
    print('\nDoing: {} (i={})'.format(grp, i))
    print('#####################\n')

    reg_tmp = linear_model.LinearRegression(fit_intercept=False)
    reg_tmp.fit(feat_vec[:,:6], dg_bind) 

    this_v = vals_v[i]
    this_B = vals_B[i]
    A = this_B.T

    h_pq = np.dot(alpha, this_v)
    alphaprime = np.dot(this_B.T, alpha)

    print('  H(p,q): {}'.format(h_pq))
    print('  aprime: {}'.format(alphaprime))

    act_h_pq = h_pq.subs([(a1, act_alpha[0]), (a2, act_alpha[1]), (a3, act_alpha[2]), (pq, 36), (p, 6), (q, 6)]) 
    act_alphaprime = np.dot(A, act_alpha)


    this_indices = np.append([0,1,2], np.in1d(x,grp).nonzero()[0]+3)
    this_feat = feat_vec[0, this_indices]

    reg_tmp.coef_[:3] = reg_inter.coef_
    reg_tmp.coef_[3:] = act_alphaprime

    pred = reg_tmp.predict(this_feat.reshape(1,-1)).item() + act_h_pq
    print('  \n(pred: {:0.2f})\n'.format(pred))

