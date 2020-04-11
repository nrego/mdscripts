from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
import logging
import shutil
import MDAnalysis

from IPython import embed

from scipy.spatial import cKDTree
import itertools
#from skimage import measure

from rhoutils import rho, cartesian
from mdtools import ParallelTool

from constants import SEL_SPEC_HEAVIES, SEL_SPEC_HEAVIES_NOWALL
from mdtools.fieldwriter import RhoField
import sys
import argparse, os
from scipy.interpolate import interp2d

import matplotlib as mpl
mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 30})

# height of cyl probe, in A
w = 9
xmin = 28.0
ymin = 5.0
zmin = 5.0

xmax = 38.5
ymax = 65.0
zmax = 65.0

box_vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

avg_n0 = np.load("cube_data_equil.dat.npz")['n0'].item()
rho_0 = avg_n0 / box_vol
rho_vols = np.load("rho_vols.dat.npy") 
expt_waters = rho_0 * rho_vols

print("{:.2f} waters in V ({} A); density: {:.4e}\n".format(avg_n0, box_vol, rho_0))

ds = np.load('rhoz_final.dat.npz')

rhoz = ds['rhoz']
xvals = ds['xvals']
rvals = ds['rvals']
xx = ds['xx']
rr = ds['rr']
max_idx = ds['max_idx'] # Index of beta phi star, for *cyl* vol, v
beta_phi_vals = ds['beta_phi_vals']
beta_phi_star = beta_phi_vals[max_idx]

print("beta phi * (for cylv v): {:.2f}".format(beta_phi_star))

# At bphistar=0.7

# Convert to nm, shift x (z) vals to bottom of small v cyl probe vol
rr = rr / 10.0
xx = (xx - 28.5) / 10.0

## Isocontour options for plotting
levels = np.linspace(0,1,3)
norm = plt.Normalize(0,1)
cmap = 'bwr_r'

homedir = os.environ['HOME']

def plot_it(idx):
    this_rho = rhoz[idx] / expt_waters


    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,6))

    pc = ax.pcolormesh(rr, xx, this_rho.T, cmap=cmap, norm=norm, shading='gourand')
    fig.colorbar(pc)
    ax.contour(rr, xx, this_rho.T, levels=[0.5], color='k')
    
    # z
    ax.set_ylim(0, w/10.+0.1)
    # R
    ax.set_xlim(0, 3.0) 
    ax.plot([0,3], [w/10.,w/10.], 'g--', linewidth=4)
    ax.set_xlabel(r'$r$ (nm)')
    ax.set_ylabel(r'$z$ (nm)')

    # Label beta phi star
    if idx == max_idx:
        label = r'$\beta \phi^*={:.2f}$'.format(beta_phi_vals[idx])
        ax.set_title(label)

        plt.tight_layout()
        plt.savefig('{}/Desktop/snap_{:03d}'.format(homedir, idx))

        [shutil.copy('{}/Desktop/snap_{:03d}.png'.format(homedir, idx), '{}/Desktop/snap_{:03d}_{:02d}.png'.format(homedir, idx, i)) for i in range(10)]
    else:
        label = r'$\beta \phi={:.2f}$'.format(beta_phi_vals[idx])
        ax.set_title(label)

        plt.tight_layout()
        plt.savefig('{}/Desktop/snap_{:03d}'.format(homedir, idx))

for i in range(max_idx+20):
    plot_it(i)




