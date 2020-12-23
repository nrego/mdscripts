
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

myorange = myorange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1] 

Rv = 20
w = 9


homedir = os.environ['HOME']


mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 30})

#fn_sph = lambda pts, x0 : 

## Analyze (reweighted) rho(x,y,z) profiles with beta phi, n
#
#  Find/linearly interpolate points of isosurface where rho(x,y,z) < s=0.5
#    Finally, fit a circle to the isopoints.

# Find xs, point where line between (xlo, ylo) and (xhi, yhi) crosses ys
def interp1d(xlo, xhi, ylo, yhi, ys=0.5):

    m = (yhi - ylo) / (xhi - xlo)

    if m == 0:
        return xlo

    return xlo + (ys-ylo)/m


# Given a density field (shape: (xvals.size, yvals.size, zvals.size)), find 
#     linearly interpolated points where we cross isovalue
def get_interp_points(rho, xvals, yvals, zvals, iso=0.5):


    rho_mask = (rho > iso).astype(int)
    xcross = np.diff(rho_mask, axis=0).astype(bool)
    ycross = np.diff(rho_mask, axis=1).astype(bool)
    zcross = np.diff(rho_mask, axis=2).astype(bool)

    dx, dy, dz = np.gradient(rho_mask)

    pts = []
    for ix in range(xvals.size-1):
        xlo = xvals[ix]
        xhi = xvals[ix+1]

        for iy in range(yvals.size-1):
            ylo = yvals[iy]
            yhi = yvals[iy+1]

            for iz in range(zvals.size-1):
                zlo = zvals[iz]
                zhi = zvals[iz+1]

                bxcross = xcross[ix, iy, iz]
                bycross = ycross[ix, iy, iz]
                bzcross = zcross[ix, iy, iz]

                if not (bxcross or bycross or bzcross):
                    continue

                ptx = interp1d(xlo, xhi, rho[ix, iy, iz], rho[ix+1, iy, iz], ys=iso) if bxcross else xlo
                pty = interp1d(ylo, yhi, rho[ix, iy, iz], rho[ix, iy+1, iz], ys=iso) if bycross else ylo
                ptz = interp1d(zlo, zhi, rho[ix, iy, iz], rho[ix, iy, iz+1], ys=iso) if bzcross else zlo

                pts.append(np.array([ptx, pty, ptz]))

    # last col of x
    for iy in range(yvals.size-1):
        ylo = yvals[iy]
        yhi = yvals[iy+1]

        for iz in range(zvals.size-1):
            zlo = zvals[iz]
            zhi = zvals[iz+1]

            bycross = ycross[-1, iy, iz]
            bzcross = zcross[-1, iy, iz]

            if not (bycross or bzcross):
                continue

            pty = interp1d(ylo, yhi, rho[-1, iy, iz], rho[-1, iy+1, iz], ys=iso) if bycross else ylo
            ptz = interp1d(zlo, zhi, rho[-1, iy, iz], rho[-1, iy, iz+1], ys=iso) if bzcross else zlo

            pts.append(np.array([xvals[-1], pty, ptz]))

    # last col of y
    for ix in range(xvals.size-1):
        xlo = xvals[ix]
        xhi = xvals[ix+1]

        for iz in range(zvals.size-1):
            zlo = zvals[iz]
            zhi = zvals[iz+1]

            bxcross = xcross[ix, -1, iz]
            bzcross = zcross[ix, -1, iz]

            if not (bxcross or bzcross):
                continue

            ptx = interp1d(xlo, xhi, rho[ix, -1, iz], rho[ix+1, -1, iz], ys=iso) if bxcross else xlo
            ptz = interp1d(zlo, zhi, rho[ix, -1, iz], rho[ix, -1, iz+1], ys=iso) if bzcross else zlo

            pts.append(np.array([ptx, yvals[-1], ptz]))

    # last col of z
    for ix in range(xvals.size-1):
        xlo = xvals[ix]
        xhi = xvals[ix+1]

        for iy in range(yvals.size-1):
            ylo = yvals[iy]
            yhi = yvals[iy+1]

            bxcross = xcross[ix, iy, -1]
            bycross = ycross[ix, iy, -1]

            if not (bxcross or bycross):
                continue

            ptx = interp1d(xlo, xhi, rho[ix, iy, -1], rho[ix+1, iy, -1], ys=iso) if bxcross else xlo
            pty = interp1d(ylo, yhi, rho[ix, iy, -1], rho[ix, iy+1, -1], ys=iso) if bycross else ylo

            pts.append(np.array([ptx, pty, zvals[-1]]))


    return np.array(pts)


# Transform a list of (x,y,z) points to (r, x) where r=sqrt( (y-y0)**2 + (z-z0)**2 )
def pts_to_rhorx(pts, y0, z0):

    ret = np.zeros((pts.shape[0], 2))

    r_sq = (pts[:,1]-y0)**2 + (pts[:,2]-z0)**2

    ret[:,0] = np.sqrt(r_sq)
    ret[:,1] = pts[:,0]

    return ret

y0=35.0
z0=35
xc = 28.5
p0 = np.array([30,10])

bounds = ([0, -np.inf], [np.inf, xc])

## Sum of squared errors for points, given sphere w/ radius R centered at (x0,y0,z0)
def x_fit(pts_yz, R, x0):
    theta = np.arccos((xc-x0)/R)

    mask = ((pts_yz[:,0] - y0)**2 + (pts_yz[:,1] - z0)**2) < (R*np.sin(theta))**2
    #mask = np.ones(pts_yz.shape[0]).astype(bool)
    ret = np.zeros(pts_yz.shape[0])

    try:
        ret[~mask] = xc
    except ValueError:
        pass
    try:
        ret[mask] = x0 + np.sqrt(R**2 - (pts_yz[mask,0] - y0)**2 - (pts_yz[mask,1] - z0)**2)
    except ValueError:
        pass


    return ret

def x_fit_noplane(pts_yz, R, x0):

    ret = np.zeros(pts_yz.shape[0])

    ret = x0 + np.sqrt(R**2 - (pts_yz[:,0] - y0)**2 - (pts_yz[:,1] - z0)**2)

    return ret

## Any points (in y, z) that have a radial distance from y0,z0 greater than buffer*RsinTheta
radial_buffer = 0.8

rvals = np.arange(0, 31, 1)
y0 = 35.0
z0 = 35.0

vals = np.arange(10, 61, 1)
yy, zz = np.meshgrid(vals, vals, indexing='ij')

univ_fit = MDAnalysis.Universe.empty(n_atoms=yy.size, trajectory=True)

## Set all other points to this, so we have a base line
cent_point = np.array([28.5, 35, 35])

rho0 = np.load('rho0.dat.npz')['rho0']


#ds_bphi = np.load('rho_bphi.dat.npz')
ds_n = np.load('rho_final.dat.npz')
#ds = np.load('rhoxyz_dx_10.dat.npz')
xbins = ds_n['xbins']
ybins = ds_n['ybins']
zbins = ds_n['zbins']

#assert np.array_equal(xbins, ds_n['xbins'])
#assert np.array_equal(ybins, ds_n['ybins'])
#assert np.array_equal(zbins, ds_n['zbins'])

xvals = xbins[:-1] + 0.5*np.diff(xbins)
yvals = ybins[:-1] + 0.5*np.diff(ybins)
zvals = zbins[:-1] + 0.5*np.diff(zbins)


#rho_bphi = ds_bphi['rho_bphi']
#assert rho_bphi.shape[1:] == rho0.shape
#beta_phi_vals = ds_bphi['beta_phi_vals']

rho_n = ds_n['rho_n']
nvals = ds_n['nvals'].astype(float)
assert rho_n.shape[1:] == rho0.shape

dx = np.diff(xvals)[0]
dy = np.diff(yvals)[0]
dz = np.diff(zvals)[0]

expt_waters = 0.033 * dx*dy*dz

## Change rho0 to simply expt number of waters per voxel
#rho0[:] = expt_waters
rho0 = 0.033

## GO THROUGH BPHI VALS ##
##########################


## GO THROUGH N VALS ##
##########################

max_atms = -np.inf
for i, nval in enumerate(nvals):

    rho = rho_n[i]
    avg_rho = rho / rho0

    mask_rho = (avg_rho > 0.5).astype(int)

    pts = get_interp_points(avg_rho, xvals, yvals, zvals)

    if pts.shape[0] > max_atms:
        max_atms = pts.shape[0]

univ = MDAnalysis.Universe.empty(n_atoms=max_atms, trajectory=True)


# Radii for each n
r_n = np.zeros_like(nvals)
# initial point x0 for each nval
x0_n = np.zeros_like(nvals)
# Contact angle with yz plane at x=28.5
theta_n = np.zeros_like(nvals)

mse_n = np.zeros_like(nvals)
r2_n = np.zeros_like(nvals)

with MDAnalysis.Writer("traj_n.xtc", univ.atoms.n_atoms) as W:

    for i, nval in enumerate(nvals):
        print("doing n: {}".format(nval))
        univ.atoms.positions[:] = 0
        rho = rho_n[i]
        avg_rho = rho / rho0

        mask_rho = (avg_rho > 0.5).astype(int)

        pts = get_interp_points(avg_rho, xvals, yvals, zvals)

        tmp_pos = univ.atoms.positions.copy()

        tmp_pos[:pts.shape[0]] = pts
        tmp_pos[pts.shape[0]:] = cent_point

        univ.atoms.positions = tmp_pos

        W.write(univ.atoms)

        if i == 0:
            univ.atoms.write("base_n.gro")

        new_p0 = p0.copy()
        max_dist = np.ceil(np.sqrt((pts[:,1] - y0)**2 + (pts[:,2] - z0)**2).max()) + 1
        if max_dist > new_p0[0]:
            new_p0[0] = max_dist

        ## Initial fit
        ## Now fit and find extent of spherical cap (e.g., y0+-RsinTheta, or z0+-RsinTheta)
        res = curve_fit(x_fit, pts[:,1:], pts[:,0], p0=new_p0, bounds=bounds)
        
        # Radius and center point x0
        R, x0 = res[0]

        h = xc - x0
        theta = 180*np.arccos(h/R)/np.pi

        ## Second fit, excluding all y, z points that are further than 
        mask_radial_xy = ((pts[:,1] - y0)**2 + (pts[:,2] - z0)**2) < radial_buffer*(R * np.sin((theta/180)*np.pi))**2
        try:
            res = curve_fit(x_fit_noplane, pts[mask_radial_xy,1:], pts[mask_radial_xy,0], p0=new_p0, bounds=bounds)
        except ValueError:
            pass
            
        R, x0 = res[0]

        ## Find rmse of fit
        pred = x_fit_noplane(pts[mask_radial_xy,1:], R, x0)
        mse = np.mean((pred - pts[mask_radial_xy,0])**2)
        r2 = 1 - (mse/pts[mask_radial_xy,0].var())

        h = xc - x0
        theta = 180*np.arccos(h/R)/np.pi

        r_n[i] = R
        x0_n[i] = x0
        theta_n[i] = theta


        mse_n[i] = mse
        r2_n[i] = r2


        ## FIT IT
        # Project iso points to r,x
        ret = pts_to_rhorx(pts, 35, 35)
        x_of_r = x_fit(np.vstack((rvals+y0, np.ones_like(rvals)*z0)).T, R, x0)
        plt.close('all')
        ax = plt.gca()

        ax.set_xlabel(r'$r$ (nm)')
        ax.set_ylabel(r'$z$ (nm)')
        ax.set_ylim(0, w/10.+0.1)
        # R
        ax.set_xlim(0, (Rv/10.)+0.4) 

        ax.plot([0,Rv/10.], [w/10.,w/10.], '-', color=myorange, linewidth=10)
        ax.axvline(Rv/10., ymin=0, ymax=(w/10.)/ax.get_ylim()[1], linewidth=10, linestyle='-', color=myorange)

        ax.plot(ret[:,0]/10.0, ret[:,1]/10.0-2.85, 'yx')
        ax.plot(rvals/10., x_of_r/10.-2.85, 'k--', linewidth=4)

        label = r'$N={};  \theta={:.2f}$'.format(nval, theta)
        ax.set_title(label)

        plt.tight_layout()

        n_idx = int(nvals.max() - nval)
        plt.savefig("/Users/nickrego/Desktop/fig_n_{:03d}".format(n_idx))

plt.close()
plt.plot(nvals, theta_n, 'x')
plt.xlabel(r'$N$')
plt.ylabel(r'$\theta$')

np.savetxt('theta_v_n.dat', np.vstack((nvals, theta_n)).T)
np.savetxt('mse_v_n.dat', np.vstack((nvals, mse_n, r2_n)).T)

'''

max_atms = -np.inf
for i, bphi in enumerate(beta_phi_vals):

    rho = rho_bphi[i]
    #avg_rho = rho / (0.033*dx*dy*dz)
    avg_rho = rho / rho0
    #avg_rho = np.clip(avg_rho, 0, 1)
    mask_rho = (avg_rho > 0.5).astype(int)

    pts = get_interp_points(avg_rho, xvals, yvals, zvals)

    if pts.shape[0] > max_atms:
        max_atms = pts.shape[0]

univ = MDAnalysis.Universe.empty(n_atoms=max_atms, trajectory=True)

# Radii for each bphi
r_bphi = np.zeros_like(beta_phi_vals)
# initial point x0 for each bphi
x0_bphi = np.zeros_like(beta_phi_vals)
# Contact angle with yz plane at x=28.5
theta_bphi = np.zeros_like(beta_phi_vals)
mse_bphi = np.zeros_like(beta_phi_vals)
r2_bphi = np.zeros_like(beta_phi_vals)

with MDAnalysis.Writer("traj_bphi.xtc", univ.atoms.n_atoms) as W:

    for i, bphi in enumerate(beta_phi_vals):
        print("doing bphi: {:.2f}".format(bphi))
        univ.atoms.positions[:] = 0
        rho = rho_bphi[i]
        avg_rho = rho / rho0

        mask_rho = (avg_rho > 0.5).astype(int)

        pts = get_interp_points(avg_rho, xvals, yvals, zvals)

        tmp_pos = univ.atoms.positions.copy()

        tmp_pos[:pts.shape[0]] = pts
        tmp_pos[pts.shape[0]:] = cent_point

        univ.atoms.positions = tmp_pos

        W.write(univ.atoms)

        if i == 0:
            univ.atoms.write("base_bphi.gro")

        new_p0 = p0.copy()
        max_dist = np.ceil(np.sqrt((pts[:,1] - y0)**2 + (pts[:,2] - z0)**2).max()) + 1
        if max_dist > new_p0[0]:
            new_p0[0] = max_dist

        ## Initial fit
        ## Now fit and find extent of spherical cap (e.g., y0+-RsinTheta, or z0+-RsinTheta)
        res = curve_fit(x_fit, pts[:,1:], pts[:,0], p0=new_p0, bounds=bounds)
        
        # Radius and center point x0
        R, x0 = res[0]

        h = xc - x0
        theta = 180*np.arccos(h/R)/np.pi

        ## Second fit, excluding all y, z points that are further than 
        mask_radial_xy = ((pts[:,1] - y0)**2 + (pts[:,2] - z0)**2) < radial_buffer*(R * np.sin((theta/180)*np.pi))**2
        res = curve_fit(x_fit_noplane, pts[mask_radial_xy,1:], pts[mask_radial_xy,0], p0=new_p0, bounds=bounds)

        R, x0 = res[0]

        ## Find rmse of fit
        pred = x_fit_noplane(pts[mask_radial_xy,1:], R, x0)
        mse = np.mean((pred - pts[mask_radial_xy,0])**2)
        r2 = 1 - (mse/pts[mask_radial_xy,0].var())

        h = xc - x0
        theta = 180*np.arccos(h/R)/np.pi

        r_bphi[i] = R
        x0_bphi[i] = x0
        theta_bphi[i] = theta

        mse_bphi[i] = mse
        r2_bphi[i] = r2

        ## FIT IT
        # Project iso points to r,x
        ret = pts_to_rhorx(pts, 35, 35)
        x_of_r = x_fit(np.vstack((rvals+y0, np.ones_like(rvals)*z0)).T, R, x0)
        plt.close('all')
        ax = plt.gca()

        ax.set_xlabel(r'$r$ (nm)')
        ax.set_ylabel(r'$z$ (nm)')
        ax.set_ylim(0, w/10.+0.1)
        # R
        ax.set_xlim(0, (Rv/10.)+0.4) 

        ax.plot([0,Rv/10.], [w/10.,w/10.], '-', color=myorange, linewidth=10)
        ax.axvline(Rv/10., ymin=0, ymax=(w/10.)/ax.get_ylim()[1], linewidth=10, linestyle='-', color=myorange)

        ax.plot(ret[:,0]/10.0, ret[:,1]/10.0-2.85, 'yx')
        ax.plot(rvals/10., x_of_r/10.-2.85, 'k--', linewidth=4)

        label = r'$\beta \phi={:.2f};  \theta={:.2f}$'.format(bphi, theta)
        ax.set_title(label)

        plt.tight_layout()

        plt.savefig("/Users/nickrego/Desktop/fig_bphi_{:03d}".format(int(bphi*100)))
'''