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

from work_managers.environment import default_env

def pbc(pos, box_dim):

    for i, box_len in enumerate(box_dim):
        pos[pos[:,i] > box_len, i] -= box_len
        pos[pos[:,i] < 0, i] += box_len

    return pos


# Get instantaneous density profile in x,y,z voxels
#
# Returns a 3d array of shape: (xvals.size-1, yvals.size-1, zvals.size-1)

def get_rhoxyz_simple(water_pos, xvals, yvals, zvals, this_idx=-1):
    
    bounds = [xvals, yvals, zvals]

    return this_idx, np.histogramdd(water_pos, bounds)[0]

# Get instantaneous density profile in x,y,z voxels
#
# Returns a 3d array of shape: (xvals.size-1, yvals.size-1, zvals.size-1)
def get_rhoxyz(water_pos, tree_grid, nx, ny, nz, cutoff=7.0, sigma=2.4, this_idx=-1):
    cutoff_sq = cutoff**2
    sigma_sq = sigma**2

    tree_water = cKDTree(water_pos)
    # Len: (n_waters,)
    #    Gives indices of grid points w/in cutoff of each water
    res = tree_water.query_ball_tree(tree_grid, r=cutoff)

    # Coarse-grained density at each grid point
    #   Shape: (n_gridpts)
    this_rho = np.zeros(tree_grid.data.shape[0])

    for idx in range(water_pos.shape[0]):

        this_water_pos = water_pos[idx]

        # Indices of all grid points within cutoff of this water
        indices = res[idx]
        if len(indices) == 0:
            continue
        #assert len(indices)

        close_gridpts = tree_grid.data[indices]

        dist_vec = close_gridpts - this_water_pos

        this_rho[indices] += rho(dist_vec.astype(np.float32), sigma, sigma_sq, cutoff, cutoff_sq)

    return this_idx, this_rho.reshape((nx, ny, nz))




extract_float_from_str = lambda instr: np.array(instr.split(), dtype=float)

parser = argparse.ArgumentParser('Output cavity voxels for each frame', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--top', type=str, default='ofile.gro', help='input structure file')
parser.add_argument('-f', '--traj', type=str, default='ofile.xtc', help='Input trajectory')
parser.add_argument('-b', '--start', default=500, type=int, help='start time, in ps')
parser.add_argument('--equil-vals', type=str, 
                    help='path to file with equilibrium values - will calc density')
parser.add_argument('-dx', default=0.5, type=float, help='spacing in x (z), in **Angstroms**. ')
parser.add_argument('-dy', default=0.5, type=float, help='spacing in y, in **angstroms**.')
parser.add_argument('-dz', default=0.5, type=float, help='spacing in y, in **angstroms**.')
#parser.add_argument('-dr', default=0.5, type=float, help='spacing in r, in **Angstroms**. ')
parser.add_argument('--rmax', default=30, type=float, help='Maximum distance r (in A), to calc rho(z,r)')
parser.add_argument('--V-min', default="28.5 10.0 10.0", type=str,
                    help="Coordinates of big V's minimum, inputted as a string of three values")
parser.add_argument('--V-max', default="48.5 60.0 60.0", type=str,
                    help="Coordinates of big V's minimum, inputted as a string of three values")
parser.add_argument('--print-out', action='store_true',
                    help='If true, print out shifted data')

default_env.add_wm_args(parser)

args = parser.parse_args()

default_env.process_wm_args(args)
wm = default_env.make_work_manager()

wm.startup()

do_calc_rho = False


## Hard (nooo!) coded dimensions of big probe V (large cubic box)
xmin, ymin, zmin = extract_float_from_str(args.V_min)
xmax, ymax, zmax = extract_float_from_str(args.V_max)

print('Vmin: ({:.2f} {:.2f} {:.2f}) Vmax: ({:.2f} {:.2f} {:.2f})'.format(xmin, ymin, zmin, xmax, ymax, zmax))
sys.stdout.flush()

box_vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

ycom = ymin + (ymax - ymin)/2.0
zcom = zmin + (zmax - zmin)/2.0
box_com = np.array([xmin, ycom, zcom])

# Equil vals avail for V (i.e., N_V and COM at equil - well, unbiased)
#.   Since they're available - calculate rho(z,r)
if args.equil_vals is not None:

    do_calc_rho = True

    equil_ds = np.load(args.equil_vals)
    # Average num waters in V at bphi=0
    avg_0 = equil_ds['n0'].item()
    # Average COM of waters in V at bphi=0
    com_0 = equil_ds['avg_com']
    # Average water density in box (at bphi=0)
    bulk_rho = avg_0 / box_vol

    dx = args.dx
    xvals = np.arange(xmin, xmax+dx, dx)
    dy = args.dy
    yvals = np.arange(ymin, ymax+dy, dy)    
    dz = args.dz
    zvals = np.arange(zmin, zmax+dz, dz)

    nx = xvals.size-1
    ny = yvals.size-1
    nz = zvals.size-1

    print("Doing rho calculation (and shifting cav COM)...")
    print("dx: {:0.2f} (from {:.2f} to {:.2f})".format(dx, xmin, xmax))
    print("dy: {:0.2f} (from {:.2f} to {:.2f})".format(dy, ymin, ymax))
    print("dz: {:0.2f} (from {:.2f} to {:.2f})".format(dz, zmin, zmax))

    xx, yy, zz = np.meshgrid(xvals[:-1], yvals[:-1], zvals[:-1], indexing='ij')
    # Center point of each voxel of V
    gridpts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T + 0.5*np.array([dx,dy,dz])
    n_voxels = gridpts.shape[0]

    tree_grid = cKDTree(gridpts)

cutoff = 7.0
sigma = 2.4

univ = MDAnalysis.Universe(args.top, args.traj)
start_frame = int(args.start / univ.trajectory.dt)
n_frames = univ.trajectory.n_frames

n_waters = np.zeros(n_frames-start_frame)
water_com = np.zeros((n_frames-start_frame, 3))

if do_calc_rho:
    # Shape: (n_frames, n_x, n_y, n_z)
    rho_xyz = np.zeros((n_frames-start_frame, nx, ny, nz), dtype=np.float32)

if args.print_out:
    W = MDAnalysis.Writer("shift.xtc", univ.atoms.n_atoms)


futures = []

for i, i_frame, in enumerate(np.arange(start_frame, n_frames)):
    if i_frame % 100 == 0:
        print("frame: {}".format(i_frame))
        sys.stdout.flush()

    univ.trajectory[i_frame]

    ## Find waters in V at this step
    waters = univ.select_atoms("name OW")
    water_pos = waters.positions

    selx = (water_pos[:,0] >= xmin) & (water_pos[:,0] < xmax)
    sely = (water_pos[:,1] >= ymin) & (water_pos[:,1] < ymax)
    selz = (water_pos[:,2] >= zmin) & (water_pos[:,2] < zmax)

    sel_mask = selx & sely & selz

    # Waters in V at this frame
    this_waters = waters[sel_mask]
    #this_waters.write("noshift.gro")

    ## COM of waters in V
    this_water_com = this_waters.positions.mean(axis=0)
    water_com[i] = this_water_com

    ## N_V: number of waters in V
    this_n_waters = this_waters.n_atoms
    n_waters[i] = this_n_waters


    ## Center cavity COM in V's COM (in y,z)
    if do_calc_rho:
        
        n_cav = avg_0 - this_n_waters
        #First, find com of cavity
        # Found from a weighted difference of water COM at bphi=0 and at this ensemble
        cavity_com = (avg_0*com_0 - this_n_waters*this_water_com) / (avg_0 - this_n_waters)
        
        # Assume no cav if sys has lost fewer than 1 % of its waters
        if (n_cav / avg_0) < 0.01 or cavity_com.min() < 0:
            print("no cav; not centering...")
            cavity_com = box_com

        # now shift all atoms so cav COM lies in center of cubic box - but only in y,z
        shift_vector = np.array([0,ycom,zcom]) - cavity_com
        shift_vector[0] = 0
        #print("frame: {} shift: {}".format(i_frame, shift_vector))
        # Shift *all* water positions in this frame 
        water_pos_shift = water_pos + shift_vector
        
        # Fix any waters that have been shifted outside of the box
        pbc(water_pos_shift, univ.dimensions[:3])
        waters.positions = water_pos_shift
        
        # Find waters that are in V after shifting cavity C.O.M.
        selx = (water_pos_shift[:,0] >= xmin-cutoff) & (water_pos_shift[:,0] < xmax+cutoff)
        sely = (water_pos_shift[:,1] >= ymin-cutoff) & (water_pos_shift[:,1] < ymax+cutoff)
        selz = (water_pos_shift[:,2] >= zmin-cutoff) & (water_pos_shift[:,2] < zmax+cutoff)

        sel_mask = selx & sely & selz

        # Waters that are in V, after shifting cav COM
        this_waters_shift = waters[sel_mask]
        
        # Finally - get instantaneous (un-normalized) density
        #   (Get rho z is *count* of waters at each x,r and x+dx,r+dr)
        #this_rho_xyz = get_rhoxyz(this_waters_shift.positions, xvals, yvals, zvals)
        #this_rho_xyz = get_rhoxyz(this_waters_shift.positions, tree_grid, nx, ny, nz, cutoff=cutoff, sigma=sigma)

        #rho_xyz[i, ...] = this_rho_xyz
        #del this_rho_xyz
        fn_args = (this_waters_shift.positions, tree_grid, nx, ny, nz)
        fn_kwargs = {'cutoff': cutoff, 'sigma': sigma, 'this_idx': i}
        
        #futures.append(wm.submit(get_rhoxyz, fn_args, fn_kwargs))
        fn_args = (this_waters_shift.positions, xvals, yvals, zvals)
        fn_kwargs = {'this_idx': i}
        futures.append(wm.submit(get_rhoxyz_simple, fn_args, fn_kwargs))

        print("submitted job {}".format(i))
        ## Optionally print out shifted frame
        if args.print_out:
            W.write(univ.atoms)
            if i_frame == n_frames - 1:
                univ.atoms.write("shift.gro")

# Output number of waters in V at each frame, as well as their COM's
#   Note: *non* shifted positions - just directly from trajectory
if not do_calc_rho:
    print("saving cube data...")
    np.savetxt("phiout_cube.dat", n_waters, fmt='%3d', header='Vmin: ({:.2f} {:.2f} {:.2f}) A Vmax: ({:.2f} {:.2f} {:.2f}) A'.format(xmin, ymin, zmin, xmax, ymax, zmax))
    np.save("com_cube.dat", water_com)


## Collect results
for i, future in enumerate(wm.as_completed(futures)):
    idx, this_rho_xyz = future.get_result(discard=True)
    if i % 100 == 0:
        print("getting result {} of {}".format(i, len(futures)))
        sys.stdout.flush()
    rho_xyz[idx, ...] = this_rho_xyz
    del this_rho_xyz

if args.print_out:
    W.close()

wm.shutdown()



if do_calc_rho:
    print("...Done: Outputting results")
    np.savez_compressed("rhoxyz.dat", rho=rho_xyz, xbins=xvals, ybins=yvals, zbins=zvals, nframes=rho_xyz.shape[0])
    
