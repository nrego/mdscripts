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

plt.close('all')
homedir = os.environ['HOME']
from IPython import embed
mpl.rcParams.update({'axes.labelsize': 45})
mpl.rcParams.update({'xtick.labelsize': 35})
mpl.rcParams.update({'ytick.labelsize': 35})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})


def print_data(reg, boot_intercept, boot_coef):
    print("    inter: {:0.2f} ({:0.4f})".format(reg.intercept_, boot_intercept.std(ddof=1)))
    errs = boot_coef.std(ddof=1, axis=0)

    print("    PQ: {:0.2f} ({:0.4f})".format(reg.coef_[0], errs[0]))
    print("    P: {:0.2f} ({:0.4f})".format(reg.coef_[1], errs[1]))
    print("    Q: {:0.2f} ({:0.4f})".format(reg.coef_[2], errs[2]))
    print("    k_o: {:0.2f} ({:0.4f})".format(reg.coef_[3], errs[3]))
    print("    noo: {:0.2f} ({:0.4f})".format(reg.coef_[4], errs[4]))
    print("    noe: {:0.2f} ({:0.4f})".format(reg.coef_[5], errs[5]))


reg_coef = np.load('sam_reg_coef.npy').item()
reg_int = np.load('sam_reg_inter.npy').item()

inter = reg_int.intercept_
# for P, Q, k_o, n_oo, n_oe
coefs = np.concatenate((reg_int.coef_, reg_coef.coef_))

ds = np.load('sam_pattern_pooled.npz')
energies = ds['energies']
delta_e = ds['delta_e']
dg_bind = ds['dg_bind']
states = ds['states']
# P Q k_o n_oo n_oe k_c n_mm n_me n_mo
feat_vec = ds['feat_vec']
errs = 1 / ds['weights']


p, q, ko, n_oo, n_oe, kc, n_mm, n_me, n_mo = [col.squeeze() for col in np.split(feat_vec, indices_or_sections=9, axis=1)]

pq = p*q
p_p_q = p+q

# Fit a dummy regression and then change its coeffs
myfeat = np.vstack((pq, p, q, ko, n_oo, n_oe)).T
#myfeat2 = np.vstack((pq, p_p_q, kc, n_mm, n_me)).T
perf_mse, err1, xvals, fit, reg = fit_leave_one(myfeat, dg_bind, weights=1/errs)
#perf_mse, err2, xvals, fit, reg2 = fit_leave_one(myfeat, energies, weights=1/errs)
boot_intercept, boot_coef = fit_bootstrap(myfeat, dg_bind, weights=1/errs)
print_data(reg, boot_intercept, boot_coef)


reg.intercept_ = reg_int.intercept_
reg.coef_[:3] = reg_int.coef_
reg.coef_[3:] = reg_coef.coef_
a1, a2, a3 = reg_coef.coef_

pred = reg.predict(myfeat)

err = dg_bind - pred
elim = np.ceil(np.abs(err).max()) + 2
bins = np.arange(-elim, elim, 0.5)

hist1, bins = np.histogram(err1, bins=bins, normed=True)
hist, bins = np.histogram(err, bins=bins, normed=True)


fig = plt.figure(figsize=(7,6))

ax = fig.gca()
ax.plot(bins[:-1], hist, label='stitched together model')
ax.plot(bins[:-1], hist1, label='model fit on all simultaneously')
plt.legend()
fig.tight_layout()
plt.savefig('{}/Desktop/fig_err_comp'.format(homedir))


plt.close('all')

np.save('sam_reg_total', reg)



