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

# Fits a circle, centered at (r,x) = (0,a), with radius b, to: r = sqrt( b2 - (x - a)^2 )
fn_fit = lambda xvals, a, b: np.sqrt(b**2 - (xvals-a)**2)

param_lb = np.array([-np.inf, 0])
param_ub = np.array([0, np.inf])
bounds = (param_lb, param_ub)
p0 = np.array([-1, 2])

def get_contour_verts(cn):
    contours = []

    # for each contour line
    assert len(cn.collections) == 1
    segs = cn.allsegs[0]

    max_seg = None
    max_len = 0
    # for each separate section of the contour line
    for seg in segs:
        if seg.shape[0] > max_len:
            max_len = seg.shape[0]
            max_seg = seg

    return max_seg

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 30})

# height of cyl probe, in A
w = 3
xmin = 28.0
ymin = 5.0
zmin = 5.0

xmax = 48.5
ymax = 65.0
zmax = 65.0

box_vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

avg_n0 = np.load("cube_data_equil.dat.npz")['n0'].item()
rho_0 = avg_n0 / box_vol
rho_vols = np.load("rho_vols.dat.npy") 
expt_waters = rho_0 * rho_vols

print("{:.2f} waters in V ({} A); density: {:.4e}\n".format(avg_n0, box_vol, rho_0))

ds = np.load('rhoz_final.dat.npz')

rhoz = ds['rhoz_n']
xvals = ds['xvals']
rvals = ds['rvals']
xx = ds['xx']
rr = ds['rr']

avg_ntwid = ds['all_avg']
beta_phi_vals = ds['beta_phi_vals']
max_idx = ds['max_idx']

n_bins = ds['n_bins']
n_vals = n_bins[:-1]
print("<N_v>0: {:.2f};  <N_v>phistar: {:.2f};  bphistar: {:.2f}".format(avg_ntwid[0], avg_ntwid[max_idx], beta_phi_vals[max_idx]))

# Index of n_vals where n is closest to <N_v>phistar
max_sus_n_idx = np.abs(n_vals - avg_ntwid[max_idx]).argmin()
last_n_idx = np.abs(n_vals - avg_ntwid[0]).argmin()

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

    print("\nn val: {:.2f}".format(n_vals[idx]))
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
    ax.set_xlim(0, 3.0) 
    ax.plot([0,3], [w/10.,w/10.], 'g--', linewidth=4)
    ax.set_xlabel(r'$r$ (nm)')
    ax.set_ylabel(r'$z$ (nm)')

    # Label beta phi star
    if idx == max_sus_n_idx:
        label = r'$N={}=\langle N_v \rangle_{{\phi^*}}$'.format(n_vals[idx])
        ax.set_title(label)
        plt.tight_layout()

        [plt.savefig('{}/Desktop/snap_{:03d}_{:02d}'.format(homedir, last_n_idx-idx, i)) for i in range(10)]
    elif idx == last_n_idx:
        label = r'$N={}=\langle N_v \rangle_0$'.format(n_vals[idx])
        ax.set_title(label)
        plt.tight_layout()
        [plt.savefig('{}/Desktop/snap_{:03d}_{:02d}'.format(homedir, last_n_idx-idx, i)) for i in range(10)]
    else:
        label = r'$N={}$'.format(n_vals[idx])
        ax.set_title(label)

        plt.tight_layout()
        plt.savefig('{}/Desktop/snap_{:03d}'.format(homedir, last_n_idx-idx))

for i, n_val in enumerate(n_vals[:last_n_idx+1]):
    plot_it(i)

