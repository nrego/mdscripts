from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

import glob, os

homedir = os.environ['HOME']

# Some shennanigans to import when running from IPython terminal
try:
    from utils import plt_errorbars
except:
    import imp
    utils = imp.load_source('utils', '{}/mdscripts/scratch/prot_hydrophobicity_comm_scripts/utils.py'.format(homedir))
    plt_errorbars = utils.plt_errorbars

mpl.rcParams.update({'axes.labelsize': 100})
mpl.rcParams.update({'xtick.labelsize': 80})
mpl.rcParams.update({'ytick.labelsize': 80})
mpl.rcParams.update({'axes.titlesize': 50})
mpl.rcParams.update({'legend.fontsize':40})

name_lup = {'1brs': 'barnase',
            '1ubq': 'ubiquitin',
            '1qgt': 'capsid subunit',
            '1ycr': 'MDM2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate dehydrogenase'}

order = ['hydrophobin', 'capsid subunit', 'MDM2', 'malate dehydrogenase', 'barnase', 'lysozyme']

fig, axes = plt.subplots(2,3, figsize=(20,11))

order_idx = []
fnames = np.array([], dtype=str)
for key, val in name_lup.items():
    if val in order:
        order_idx.append(order.index(val))
        fnames = np.append(fnames, '{}/phi_sims/ntwid_out.dat'.format(key))
fnames = fnames[np.argsort(order_idx)]

axes = axes.T.reshape(6)


for idir, fname in enumerate(fnames):

    ax = axes[idir]
    dirname = os.path.dirname(fname)
    root_dir = os.path.dirname(dirname)

    print('dir: {}'.format(root_dir))
    dat = np.loadtxt(fname)
    err = np.loadtxt('{}/ntwid_var_err.dat'.format(dirname))

    ax.errorbar(dat[:,0], dat[:,2], yerr=err, fmt='k-o', linewidth=6, elinewidth=4, capsize=5, capthick=2, markersize=12, barsabove=True)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(0, 4.1)
    ax.set_xticks([0,2,4])
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([0,500,1000,1500,2000])
    
    if ymax < 500:
        ax.set_yticks([0,250])

    ax.set_ylim(0, ymax)
    #ax.set_title("{}".format(order[idir]))
    ax.set_xlabel(r'$\beta \phi$')
    ax.set_ylabel(r'$\chi_v$')

plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/prot_comparison.pdf', transparent=True)
plt.show()

