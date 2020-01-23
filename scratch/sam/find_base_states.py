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

from scratch.sam.util import *
import itertools

plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':30})


def get_cov_rank(feat_vec):
    d = feat_vec - feat_vec.mean(axis=0)
    cov = np.dot(d.T, d) / d.shape[0]

    return np.linalg.matrix_rank(cov)



reg_coef = np.load('sam_reg_pooled.npy').item()
reg_int = np.load('sam_reg_inter.npy').item()

inter = reg_int.intercept_
# for P, Q, k_o, n_oo, n_oe
coefs = np.concatenate((reg_int.coef_, reg_coef.coef_))

ds = np.load('sam_pattern_pooled.npz')
energies = ds['energies']
states = ds['states']
feat_vec = ds['feat_vec']
errs = 1 / ds['weights']

p, q, ko, n_oo, n_oe, kc, n_mm, n_me, n_mo = np.split(feat_vec, indices_or_sections=9, axis=1)

pq = p*q
p_p_q = p+q

feat_vec = np.hstack((pq, p_p_q, ko, n_oo, n_oe, kc, n_mm, n_me, n_mo))
rank = get_cov_rank(feat_vec)

avail_indices = np.arange(2, 9)

coef_names = np.array(['PQ', 'P+Q', 'ko', 'noo', 'noe', 'kc', 'ncc', 'nce', 'nco'])

for idx in itertools.combinations(avail_indices, rank-2):
    
    this_indices = np.append((0,1), idx)
    this_feat = feat_vec[:,this_indices]
    this_rank = get_cov_rank(this_feat)

    if this_rank < rank:
        continue

    print(coef_names[np.array(idx)])