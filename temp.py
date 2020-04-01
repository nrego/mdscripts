from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
import logging

import MDAnalysis
#from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

from IPython import embed

from scipy.spatial import cKDTree
import itertools
#from skimage import measure

from rhoutils import rho, cartesian
from mdtools import ParallelTool

from constants import SEL_SPEC_HEAVIES, SEL_SPEC_HEAVIES_NOWALL
from mdtools.fieldwriter import RhoField
import sys
import argparse


def get_rhoz(water_pos, box_com, xvals, rvals, bulk_rho):
    # Shape: (n_xvals-1, n_rvals-1)
    max_n = 0
    rhoz = np.zeros((xvals.size-1, rvals.size-1))
    # R2 = y**2 + z**2
    water_distances = np.sqrt(((water_pos-box_com)**2)[:,1:].sum(axis=1))

    for ix, xval_lb in enumerate(xvals[:-1]):
        xval_lb = np.round(xval_lb, 5)
        xval_ub = np.round(xvals[ix+1], 5)

        this_dx = xval_ub - xval_lb

        ## Mask for waters that are between xval_lb and xval_ub
        xmask = (water_pos[:,0] >= xval_lb) & (water_pos[:,0] < xval_ub)
        x_water_pos = water_pos[xmask]

        for ir, rval_lb in enumerate(rvals[:-1]):
            rval_lb = np.round(rval_lb, 5)
            rval_ub = np.round(rvals[ir+1], 5)

            this_dr = rval_ub - rval_lb

            this_vol = this_dx*np.pi*(rval_ub**2 - rval_lb**2)
            expt_waters = this_vol * bulk_rho
            ## Mask for waters that are between rval_lb and rval_ub in y,z
            rmask = (water_distances >= rval_lb) & (water_distances < rval_ub)

            tot_mask = xmask & rmask

            if tot_mask.sum() > max_n:
                max_n = tot_mask.sum()
            rhoz[ix, ir] = tot_mask.sum() #/ expt_waters

    return rhoz


parser = argparse.ArgumentParser('Output cavity voxels for each frame')
parser.add_argument('-c', '--top', type=str, default='ofile.gro', help='input structure file')
parser.add_argument('-f', '--traj', type=str, default='ofile.xtc', help='Input trajectory')
parser.add_argument('-b', '--start', default=500, type=int, help='start time, in ps')
parser.add_argument('--shift-data', type=str, help='Dataset with average number of waters within cube, as well as unbiased water COM')
args = parser.parse_args()


do_shift = False

xmin = 28.0
ymin = 15.0
zmin = 15.0

xmax = 38.5
ymax = 55.0
zmax = 55.0

box_vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

if args.shift_data is not None:
    do_shift = True
    shift_ds = np.load(args.shift_data)
    avg_0 = shift_ds['n0'].item()
    com_0 = shift_ds['avg_com']
    bulk_rho = avg_0 / box_vol


ycom = ymin + (ymax - ymin)/2.0
zcom = zmin + (zmax - zmin)/2.0
box_com = np.array([xmin, ycom, zcom])

dx = 0.4
xvals = np.arange(xmin, xmax+dx, dx)
dr = 0.5
rvals = np.arange(0, 20+dr, dr)

univ = MDAnalysis.Universe(args.top, args.traj)
start_frame = int(args.start / univ.trajectory.dt)
n_frames = univ.trajectory.n_frames

n_waters = np.zeros(n_frames-start_frame)
water_com = np.zeros((n_frames-start_frame, 3))

rho_z = np.zeros((n_frames-start_frame, xvals.size-1, rvals.size-1))

for i, i_frame, in enumerate(np.arange(start_frame, n_frames)):
    if i_frame % 100 == 0:
        print("frame: {}".format(i_frame))
        sys.stdout.flush()

    univ.trajectory[i_frame]
    waters = univ.select_atoms("name OW")
    water_pos = waters.positions

    selx = (water_pos[:,0] >= xmin) & (water_pos[:,0] < xmax)
    sely = (water_pos[:,1] >= ymin) & (water_pos[:,1] < ymax)
    selz = (water_pos[:,2] >= zmin) & (water_pos[:,2] < zmax)

    sel_mask = selx & sely & selz

    this_waters = waters[sel_mask]
    #this_waters.write("noshift.gro")

    this_water_com = this_waters.positions.mean(axis=0)
    water_com[i] = this_water_com

    this_n_waters = this_waters.n_atoms
    n_waters[i] = this_n_waters

    if do_shift:
        
        #First, find com of cavity
        cavity_com = (avg_0*com_0 - this_n_waters*this_water_com) / (avg_0 - this_n_waters)
        
        # now shift all atoms so cav COM lies in center of cubic box - but only in y,z
        shift_vector = np.array([0,ycom,zcom]) - cavity_com
        shift_vector[0] = 0
        
        # Shift water positions
        water_pos_shift = water_pos + shift_vector
        waters.positions = water_pos_shift
        
        # Find waters that are still in box after shifting
        selx = (water_pos_shift[:,0] >= xmin) & (water_pos_shift[:,0] < xmax)
        sely = (water_pos_shift[:,1] >= ymin) & (water_pos_shift[:,1] < ymax)
        selz = (water_pos_shift[:,2] >= zmin) & (water_pos_shift[:,2] < zmax)
        sel_mask = selx & sely & selz

        # Waters that are in cubic box, after shifting cav COM
        this_waters_shift = waters[sel_mask]
        this_rho_z = get_rhoz(this_waters_shift.positions, box_com, xvals, rvals, bulk_rho)

        rho_z[i] = this_rho_z

np.savetxt("phiout_cube.dat", n_waters, fmt='%3d')
np.save("com_noshift.dat", water_com)
np.savez_compressed("rhoz.dat", rho_z=rho_z, xvals=xvals, rvals=rvals)
