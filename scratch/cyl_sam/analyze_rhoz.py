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
myorange = myorange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1] 
# Fits a circle, centered at (r,x) = (0,a), radius b, to: x = a + sqrt(b**2 - r**2)
fn_fit = lambda rvals, a, b: a + np.sqrt(b**2 - rvals**2)

## Parameters for circle fitting
# In (a, b); a is x pt of circle center, b is circle radius
param_lb = np.array([-np.inf, 0])
param_ub = np.array([0, np.inf])
bounds = (param_lb, param_ub)
p0 = np.array([-1, 2])


# Find xs, point where line between (xlo, ylo) and (xhi, yhi) crosses ys
def interp1d(xlo, xhi, ylo, yhi, ys=0.5):
    m = (yhi - ylo) / (xhi - xlo)

    return xlo + (ys-ylo)/m


# xvals is x bin boundaries
def get_isovals(this_rho, xvals, iso=0.5):
    n_xvals, n_rvals = this_rho.shape

    # Point, in x, at which we cross iso for each rval
    out_iso = np.zeros(n_rvals)

    for i in range(n_rvals):
        # Rho vals, in x, at this value of r
        rho_slice = this_rho[:,i]

        # Index where rho goes from under to over iso
        cross_mask = (rho_slice[:-1] <= iso) & (rho_slice[1:] > iso)
        if cross_mask.sum() > 1:
            print('multiple crosses...')

        # index of first crossing
        cross_idx = cross_mask.argmax()

        # value of x where this rho_slice crosses iso val, interpolated
        xs = interp1d(xvals[cross_idx], xvals[cross_idx+1], rho_slice[cross_idx], rho_slice[cross_idx+1], iso)
        out_iso[i] = xs

    return out_iso


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
# Radius of cylinder, in A
Rv = 20
# Box V
xmin = 28.5
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

assert np.unique(np.diff(rvals)).size == 1
assert np.unique(np.diff(xvals)).size == 1

dr = np.diff(rvals)[0]
dx = np.diff(xvals)[0]

# Adjust to be center of bins
rvals += 0.5*dr
xvals += 0.5*dx

# Convert to nm, shift x (z) vals to bottom of small v cyl probe vol
rr = rr / 10.0
xx = (xx - 28.5) / 10.0
rvals = rvals / 10.0
xvals = (xvals - 28.5) / 10.0



## Isocontour options for plotting

norm = plt.Normalize(0,1, clip=True)
cmap = 'coolwarm_r'

homedir = os.environ['HOME']




def plot_it(idx):

    print("\nBeta phi: {:.2f}".format(beta_phi_vals[idx]))
    #this_rho = rhoz[idx] / expt_waters
    this_rho = rhoz[idx] / rhoz[0]

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,6))

    pc = ax.pcolormesh(rr, xx, this_rho.T, cmap=cmap, norm=norm, shading='gourand')
    #pc = ax.contourf(rr, xx, norm(this_rho.T), levels=[0.0, 0.5, 1.0], cmap=cmap)
    fig.colorbar(pc)
    #cn = ax.contour(rr, xx, this_rho.T, levels=[0.5], color='k')
    
    # Now let's fit our contour....
    # Will be shape (rvals,); gives. the value of x at each r point where we cross below threshold
    contour = get_isovals(this_rho, xvals)
    # Mask out negative crosses (neg x, doesnt make sense) or crosses that happen too high
    mask = (contour > 0) & (contour < xvals.max()) & (rvals[:-1] < Rv/10.)
    ax.plot(rvals[:-1][mask], contour[mask], 'x', markersize=12)
    
    ## Fitted circle to contour isovals
    a, b = curve_fit(fn_fit, rvals[:-1][mask], contour[mask], p0=p0, maxfev=5000, bounds=bounds)[0]
    test_rvals = np.arange(0, rvals.max(), dr/10.)
    fitvals = fn_fit(test_rvals, a, b)
    ax.plot(rvals, fitvals, 'k--', linewidth=3)

    # z
    ax.set_ylim(0, w/10.+0.1)
    # R
    ax.set_xlim(0, (Rv/10.)+0.4) 

    # Plot bounds of cylinder
    ax.plot([0,Rv/10.], [w/10.,w/10.], 'o-', linewidth=10)
    ax.axvline(Rv/10., ymin=0, ymax=(w/10.)/ax.get_ylim()[1], linewidth=10, linestyle='-', color=myorange)
    ax.set_xlabel(r'$r$ (nm)')
    ax.set_ylabel(r'$z$ (nm)')

    ## SAVE OUT FIGURE

    # Label beta phi star
    if idx == max_idx:
        label = r'$\beta \phi^*={:.2f}$'.format(beta_phi_vals[idx])
        ax.set_title(label)

        plt.tight_layout()
        plt.savefig('{}/Desktop/snap_{:03d}'.format(homedir, idx))

        [shutil.copy('{}/Desktop/snap_{:03d}.png'.format(homedir, idx), '{}/Desktop/snap_{:03d}_{:02d}.png'.format(homedir, idx, i)) for i in range(4)]
    else:
        label = r'$\beta \phi={:.2f}$'.format(beta_phi_vals[idx])
        ax.set_title(label)

        plt.tight_layout()
        plt.savefig('{}/Desktop/snap_{:03d}'.format(homedir, idx))

    plt.close('all')

for i in range(max_idx-2, max_idx+9):
    plot_it(i)




