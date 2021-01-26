
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
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

from skimage.measure import marching_cubes_lewiner

import os, glob

myorange = myorange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1] 

Rv = 20
w = 9


homedir = os.environ['HOME']


mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 30})

## Get boundaries
ds = np.load('rho_n_0000.dat.npz')
xbins = ds['xbins']
ybins = ds['ybins']
zbins = ds['zbins']

dx = np.diff(xbins)[0]
dy = np.diff(ybins)[0]
dz = np.diff(zbins)[0]

y0 = ybins.mean()
z0 = zbins.mean()
xc = xbins.min()

min_pt = np.array([xbins.min(), ybins.min(), zbins.min()])

print("\nDoing Rv={}A w={}A".format(Rv, w))
print("min pt: {}".format(min_pt))
print("xc: {} y0: {}  z0: {}".format(xc, y0, z0))


## Initial guess for (R, x0)
p0 = np.array([30,10])

bounds = ([0, -np.inf], [np.inf, xc])

min_pt = np.array([xbins.min(), ybins.min(), zbins.min()])

rvals = np.arange(0, 31, 1)

## Set all other points to this, so we have a base line
cent_point = np.array([xc, y0, z0])



# Fit points in y,z plane to spherical cap with radius R centered at (x0,y0,z0)
def x_fit(pts_yz, R, x0):

    # Fit (that is, S(y,z)) for each y,z point
    ret = np.zeros(pts_yz.shape[0])

    if R == 0 or x0 == xc:
        ret[:] = xc
        return ret

    try:
        ret = x0 + np.sqrt(R**2 - (pts_yz[:,0] - y0)**2 - (pts_yz[:,1] - z0)**2)
    except ValueError:
        pass


    return ret

# Transform a list of (x,y,z) points to (r, x) where r=sqrt( (y-y0)**2 + (z-z0)**2 )
def pts_to_rhorx(pts, y0, z0):

    ret = np.zeros((pts.shape[0], 2))

    r_sq = (pts[:,1]-y0)**2 + (pts[:,2]-z0)**2

    ret[:,0] = np.sqrt(r_sq)
    ret[:,1] = pts[:,0]

    return ret


def process_fname(idx, fname):

    ds = np.load(fname)

    n_val = ds['n_val'].item()

    assert np.array_equal(xbins, ds['xbins'])
    assert np.array_equal(ybins, ds['ybins'])
    assert np.array_equal(zbins, ds['zbins'])

    avg_cav = ds['avg_cav']


    # Try to fit isosurf - except means no cavity
    try:
        pts, faces, norms, vals = marching_cubes_lewiner(avg_cav, level=0.5, spacing=np.array([dx,dy,dz]))
        
        pts += min_pt

        #new_p0 = p0.copy()
        #max_dist = np.ceil(np.sqrt((pts[:,1] - y0)**2 + (pts[:,2] - z0)**2).max()) + 1
        #if max_dist > new_p0[0]:
        #    new_p0[0] = max_dist
        

        res = curve_fit(x_fit, pts[:,1:], pts[:,0], p0=p0, bounds=bounds)
        R, x0 = res[0]
        h = xc - x0
        this_theta = 180*np.arccos(h/R)/np.pi

        pred = x_fit(pts[:,1:], R, x0)

        diff_sq = (pts[:,0] - pred)**2
        this_fit_mse = np.mean(diff_sq)

        mse_hat_cav = np.mean(ds['mse_hat_cav'])


        ## Project isosurf points to (r, x)
        proj_pts = pts_to_rhorx(pts, y0, z0)

        proj_fit = x_fit(np.vstack((rvals+y0, np.ones_like(rvals)*z0)).T, R, x0)

    # No cavity
    except ValueError:
        this_theta = 0
        this_fit_mse = np.nan

        mse_hat_cav = np.mean(ds['mse_hat_cav'])

        proj_fit = np.ones_like(rvals)

        proj_pts = None

    plt.close('all')
    ax = plt.gca()

    ax.set_xlabel(r'$r$ (nm)')
    ax.set_ylabel(r'$z$ (nm)')
    ax.set_ylim(0, w/10.+0.1)
    # R
    ax.set_xlim(0, (Rv/10.)+0.4)  

    ax.plot([0,Rv/10.], [w/10.,w/10.], '-', color=myorange, linewidth=10)
    ax.axvline(Rv/10., ymin=0, ymax=(w/10.)/ax.get_ylim()[1], linewidth=10, linestyle='-', color=myorange)

    if proj_pts is not None:
        ax.plot(proj_pts[:,0]/10., (proj_pts[:,1]-xc)/10., 'yx')
    ax.plot(rvals/10., (proj_fit-xc)/10., 'k--', linewidth=4)

    label = r'$N={};  \theta={:.2f}$'.format(n_val, this_theta)
    ax.set_title(label)

    plt.tight_layout()

    plt.savefig("/Users/nickrego/Desktop/fig_n_{:03d}".format(idx))


    return n_val, this_theta, mse_hat_cav, this_fit_mse


fnames = sorted(glob.glob("rho_n*"))[::-1]

n_vals = np.zeros(len(fnames))
mses_hat_cav = np.zeros_like(n_vals)
mses_fit = np.zeros_like(n_vals)
thetas = np.zeros_like(n_vals)
# At each N val, number of cavity voxels in the binary average cavity field - size of average
#    cav at this N, essentially
n_avg_cav = np.zeros_like(n_vals)

for i, fname in enumerate(fnames):
    n_val, this_theta, mse_hat_cav, this_fit_mse = process_fname(i, fname)

    print("Doing N: {}".format(n_val))

    n_vals[i] = n_val
    thetas[i] = this_theta
    mses_hat_cav[i] = mse_hat_cav
    mses_fit[i] = this_fit_mse

    n_avg_cav[i] = (np.load(fname)['avg_cav'] > 0.5).sum()


np.savetxt('theta_v_n.dat', np.vstack((n_vals, thetas)).T)
np.savetxt('mse_v_n.dat', np.vstack((n_vals, n_avg_cav, mses_hat_cav, mses_fit)).T)


