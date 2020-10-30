## Gather all tp,fp,tn,fn for all systems
#  calculate f1, etc

import matplotlib as mpl
from matplotlib import rc 
import os, glob
import numpy as np

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 50})
mpl.rcParams.update({'ytick.labelsize': 50})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

from constants import k

beta = 1/(k * 300)

sys_names = []
perf = np.array([])
tprs = np.array([])
precs = np.array([])
phis = np.array([])

dimers = ['2tsc', '1msb', '1pp2', '1ycr_mdm2', 'ubiq_merge', '1bmd', '1brs_bn']

fnames = ['{}/pred_reweight/performance_first_global.dat'.format(name) for name in dimers]

outdat = np.zeros((len(fnames), 8), dtype=object)

rc('text', usetex=False)
names = []
phi_f1 = []
phi_fh = []
dh_opts = []

for i,fname in enumerate(fnames):
    dirname = os.path.dirname(fname)
    dat = np.loadtxt(fname)

    phi, tp, fp, tn, fn, tpr, fpr, ppv, f_h, f_1, mcc = np.split(dat, 11, 1)

    best_f1 = np.argmax(f_1)
    best_fh = np.argmax(f_h)
    
    name = fname.split('/')[0]
    names.append(name)
    phi_f1.append(phi[best_f1])
    phi_fh.append(phi[best_fh])

    print("system: {}  beta_phi_opt: {:.2f}  d_h: {:.4f}".format(dimers[i], phi[best_fh].item(), f_h[best_fh].item()))
    print("  f1:  beta phi opt: {:.2f}   f_1: {:.4f}".format(phi[best_f1].item(), f_1[best_f1].item()))
    # Fraction of dewetted atoms that are phobic
    phob_frac = np.loadtxt('{}/phobicity_with_betaphi.dat'.format(dirname))
    plt.plot(phob_frac[:,0], phob_frac[:,2], 'o-', label=dimers[i])

indices = np.arange(len(names))


