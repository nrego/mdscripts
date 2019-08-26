import numpy as np

from scratch.sam.util import *
from scratch.neural_net.lib import *
from scratch.interactions.util import *

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from scipy.spatial import cKDTree

import os, glob

import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

homedir = os.environ["HOME"]
def reshape_feat(feat):
    ret = feat.reshape(6,6).T[::-1, ::-1]

    return ret.reshape(1,1,6,6)

def plt_errorbars(bb, vals, errs, **kwargs):
    ax = plt.gca()
    ax.fill_between(bb, vals-errs, vals+errs, alpha=0.5, **kwargs)


ds = np.load("sam_pattern_data.dat.npz")
feat_vec, energies, poly, beta_phi_stars, positions, patch_indices, methyl_pos, adj_mat = load_and_prep()
n_dat = feat_vec.shape[0]

fig, ax = plt.subplots(figsize=(7,6))
pvn = np.loadtxt("{}/simulations/pattern_sample/k_07/d_075/trial_0/PvN.dat".format(homedir))

ax.plot(pvn[:,0], pvn[:,1])
plt_errorbars(pvn[:,0], pvn[:,1], pvn[:,2])

fig.tight_layout()
fig.savefig("{}/Desktop/fig.png".format(homedir), transparent=True)

plt.close("all")

norm = plt.Normalize(0,1)

feat = reshape_feat(feat_vec[-1])
plot_hextensor(feat, norm=norm)
plt.savefig("{}/Desktop/k_00.png".format(homedir), transparent=True)

feat = reshape_feat(feat_vec[-2])
plot_hextensor(feat, norm=norm)
plt.savefig("{}/Desktop/k_36.png".format(homedir), transparent=True)


feat = reshape_feat(feat_vec[700])
plot_hextensor()

plt.close("all")





