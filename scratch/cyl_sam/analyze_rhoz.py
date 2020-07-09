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

from skimage import measure
from scipy.optimize import curve_fit

# Fits a circle, centered at (r,z) = (0,-a), to r2 = b2 - (z - a)^2
fn_fit = lambda rvals, a, b: a + np.sqrt(b**2 - rvals**2)
param_lb = np.array([-np.inf, 0])
param_ub = np.array([0, np.inf])
bounds = (param_lb, param_ub)
p0 = np.array([-1, 2])

def get_contour_verts(cn):
    contours = []

    # for each contour line
    assert len(cn.collections) == 1
    cc = cn.collections[0]
    paths = []
    max_len = 0
    max_path = None
    # for each separate section of the contour line
    for pp in cc.get_paths():
        xy = []
        # for each segment of that section
        for vv in pp.iter_segments():
            xy.append(vv[0])

        xy = np.vstack(xy)
        if xy.shape[0] > max_len:
            max_len = xy.shape[0]
            max_path = xy

        paths.append(np.vstack(xy))

    return max_path


mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 30})

# height of cyl probe, in A
w = 9
xmin = 28.0
ymin = 10.0
zmin = 10.0

xmax = 40.0
ymax = 60.0
zmax = 60.0

box_vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

avg_n0 = np.load("cube_data_equil.dat.npz")['n0'].item()
rho_0 = avg_n0 / box_vol
rho_vols = np.load("Equil/rho_vols.dat.npy") 
expt_waters = rho_0 * rho_vols

print("{:.2f} waters in V ({} A); density: {:.4e}\n".format(avg_n0, box_vol, rho_0))

ds = np.load('rhoz_final.dat.npz')

rhoz = ds['rhoz']
xvals = ds['xvals']
rvals = ds['rvals']
xx = ds['xx']
rr = ds['rr']
max_idx = ds['max_idx'] # Index of beta phi star, for *cyl* vol, v
#max_idx = 35
beta_phi_vals = ds['beta_phi_vals']
beta_phi_star = beta_phi_vals[max_idx]

print("beta phi * (for cylv v): {:.2f}".format(beta_phi_star))

# At bphistar=0.7
dr = np.diff(rvals)[0]
dx = np.diff(xvals)[0]
# Convert to nm, shift x (z) vals to bottom of small v cyl probe vol
rr = rr / 10.0
xx = (xx - 28.5) / 10.0
r_small = np.unique(rr) < 1.5

## Isocontour options for plotting
levels = np.linspace(0,1,3)
norm = plt.Normalize(0,1)
cmap = 'coolwarm_r'

homedir = os.environ['HOME']

def plot_it(idx):

    print("\nBeta phi: {:.2f}".format(beta_phi_vals[idx]))
    this_rho = rhoz[idx] / expt_waters

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,6))

    pc = ax.pcolormesh(rr, xx, this_rho.T, cmap=cmap, norm=norm, shading='gourand')
    fig.colorbar(pc)
    cn = ax.contour(rr, xx, this_rho.T, levels=[0.5], color='k')
    
    # Now let's fit our contour....
    contour = get_contour_verts(cn)
    mask = contour[:,0] < 1.5
    #ax.plot(contour[:,0], contour[:,1], 'x')
    a, b = curve_fit(fn_fit, contour[mask,0], contour[mask,1], p0=p0, maxfev=5000, bounds=bounds)[0]
    fitvals = fn_fit(np.unique(rr), a, b)
    ax.plot(np.unique(rr), fitvals, 'k--', linewidth=3)

    # z
    ax.set_ylim(0, w/10.+0.1)
    # R
    ax.set_xlim(0, 2.4) 
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

for i in range(max_idx-2, max_idx+3):
    plot_it(i)




