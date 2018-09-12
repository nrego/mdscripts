from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import glob, os


def plt_errorbars(bb, vals, errs, **kwargs):
    plt.fill_between(bb, vals-errs, vals+errs, alpha=0.5, **kwargs)
    
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

mpl.rcParams.update({'axes.labelsize': 100})
mpl.rcParams.update({'xtick.labelsize': 80})
mpl.rcParams.update({'ytick.labelsize': 80})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':30})

name_lup = {'1brs': 'barnase',
            '1ubq': 'ubiquitin',
            '1qgt': 'capsid',
            '1ycr': 'mdm2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate dehydrogenase'}

order = ['hydrophobin', 'capsid', 'mdm2', 'malate dehydrogenase', 'barnase', 'lysozyme']

from constants import k

beta = 1/(k*300)

#fig, axes = plt.subplots(2,3, sharex=True)

order_idx = []
fnames = np.array([], dtype=str)
for key, val in name_lup.iteritems():
    if val in order:
        order_idx.append(order.index(val))
        fnames = np.append(fnames, '{}/phi_sims/ntwid_out.dat'.format(key))
fnames = fnames[np.argsort(order_idx)]
#axes = axes.T.reshape(6)
for idir, fname in enumerate(fnames):

    fig, ax = plt.subplots()
    #ax = axes[idir]
    dirname = os.path.dirname(fname)
    root_dir = os.path.dirname(dirname)

    print('dir: {}'.format(root_dir))
    dat = np.loadtxt(fname)
    err = np.loadtxt('{}/ntwid_var_err.dat'.format(dirname))

    ax.errorbar(dat[:,0], dat[:,2], yerr=err, fmt='k-o', linewidth=12, elinewidth=5, markersize=25)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(0, beta*10)
    ax.set_xticks([0,2,4])
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([250,500,750,1000])
    ax.set_ylim(0, ymax)
    

    ax.set_xlabel(r'$\beta \phi$')
    ax.set_ylabel(r'$\chi_v$')
    plt.tight_layout()
    
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.5)
    set_size(4.5,4.5,ax)
    plt.savefig('/Users/nickrego/Desktop/{}.pdf'.format(root_dir))
    plt.show()

