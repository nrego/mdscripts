from __future__ import division, print_function

import matplotlib as mpl
from matplotlib import rc 
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from constants import k

homedir = os.environ['HOME']

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':10})

beta = 1 / (300*k)
## Plot rmsd vs phi ##
sys_names = ['2tsc', '1msb', '1pp2', '1ycr_mdm2', 'ubiq_merge', '1bi4', '1brs_bn']

name_lut = {
    '2b97': 'Hydrophobin',
    '2tsc': 'Thymidylate\nsynthase',
    '1bi4': 'HIV\nInteg',
    '1ycr_mdm2': 'MDM2',
    '1bmd': 'Malate\ndehydrogenase',
    '1msb': 'Mannose-\nbinding\nprotein',
    'ubiq_merge': 'Ubiquitin',
    '1brs_bn': 'Barnase',
    '1pp2': 'phospholipase a'
}

fig, ax = plt.subplots(figsize=(10,8))


for sys_name in sys_names:
    fnames = glob.glob('{}/pred/phi_*/rmsd_fit.dat'.format(sys_name))
    phi_vals = np.array([])
    rmsd_backbone_vals = np.array([])
    rmsd_heavy_vals = np.array([])

    for fname in fnames:
        phi = float(fname.split('/')[2].split('_')[-1]) / 10.0
        phi_vals = np.append(phi_vals, phi)

        rmsd_backbone, rmsd_heavy = np.loadtxt(fname)[1:]

        rmsd_backbone_vals = np.append(rmsd_backbone_vals, rmsd_backbone)
        rmsd_heavy_vals = np.append(rmsd_heavy_vals, rmsd_heavy)

    ax.plot(beta*phi_vals, rmsd_heavy_vals/10.0, label=name_lut[sys_name])

ax.set_xlabel(r'$\beta \phi$')
ax.set_ylabel(r'RMSD (nm)')
ax.legend()

fig.tight_layout()
fig.savefig('{}/Desktop/rmsd_v_phi.pdf'.format(homedir))



