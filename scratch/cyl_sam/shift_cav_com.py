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
from skimage.measure import marching_cubes_lewiner

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
def get_rhoxyz(water_pos, sam_pos, tree_grid, nx, ny, nz, cutoff=7.0, sigma=2.4, this_idx=-1):
    cutoff_sq = cutoff**2
    sigma_sq = sigma**2

    tree_water = cKDTree(water_pos)
    tree_sam = cKDTree(sam_pos)
    # Len: (n_waters,)
    #    Gives indices of grid points w/in cutoff of each water
    res = tree_water.query_ball_tree(tree_grid, r=cutoff)

    # Coarse-grained density at each grid point
    #   Shape: (n_gridpts)
    this_rho_water = np.zeros(tree_grid.data.shape[0])

    for idx in range(water_pos.shape[0]):

        this_water_pos = water_pos[idx]

        # Indices of all grid points within cutoff of this water
        indices = res[idx]
        if len(indices) == 0:
            continue
        #assert len(indices)

        close_gridpts = tree_grid.data[indices]

        dist_vec = close_gridpts - this_water_pos

        this_rho_water[indices] += rho(dist_vec.astype(np.float32), sigma, sigma_sq, cutoff, cutoff_sq)


    res = tree_sam.query_ball_tree(tree_grid, r=cutoff)

    this_rho_sam = np.zeros(tree_grid.data.shape[0])

    for idx in range(sam_pos.shape[0]):

        this_sam_pos = sam_pos[idx]

        # Indices of all grid points within cutoff of this sam atom
        indices = res[idx]
        if len(indices) == 0:
            continue
        #assert len(indices)

        close_gridpts = tree_grid.data[indices]

        dist_vec = close_gridpts - this_sam_pos

        this_rho_sam[indices] += rho(dist_vec.astype(np.float32), sigma, sigma_sq, cutoff, cutoff_sq)



    return this_idx, this_rho_water.reshape((nx, ny, nz)), this_rho_sam.reshape((nx, ny, nz))




extract_float_from_str = lambda instr: np.array(instr.split(), dtype=float)

parser = argparse.ArgumentParser('Output cavity voxels for each frame', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--top', type=str, default='ofile.gro', help='input structure file')
parser.add_argument('-f', '--traj', type=str, default='ofile.xtc', help='Input trajectory')
parser.add_argument('-b', '--start', default=500, type=int, help='start time, in ps')
parser.add_argument('--equil-vals', type=str, 
                    help='path to file with equilibrium values - will calc density')
parser.add_argument('-dx', default=1, type=float, help='spacing in x (z), in **Angstroms**. ')
parser.add_argument('-dy', default=1, type=float, help='spacing in y, in **angstroms**.')
parser.add_argument('-dz', default=1, type=float, help='spacing in y, in **angstroms**.')
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

#wm.startup()

do_calc_rho = False


## Hard (nooo!) coded dimensions of big probe V (large cubic box)
xmin, ymin, zmin = extract_float_from_str(args.V_min)
xmax, ymax, zmax = extract_float_from_str(args.V_max)

print('Vmin: ({:.2f} {:.2f} {:.2f}) Vmax: ({:.2f} {:.2f} {:.2f})'.format(xmin, ymin, zmin, xmax, ymax, zmax))
sys.stdout.flush()

box_vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

ycom = (ymin + ymax)/2.0
zcom = (zmin + zmax)/2.0
box_com = np.array([xmin, ycom, zcom])

## Set up voxel grid

dx = args.dx
xvals = np.arange(xmin, xmax+dx, dx)
dy = args.dy
yvals = np.arange(ymin, ymax+dy, dy)    
dz = args.dz
zvals = np.arange(zmin, zmax+dz, dz)

expt_waters = 0.033 #* dx * dy * dz
expt_sam = 0.0384 #* dx * dy * dz

nx = xvals.size-1
ny = yvals.size-1
nz = zvals.size-1

print("Doing rho calculation (and shifting cav COM)...")
print("dx: {:0.2f} (from {:.2f} to {:.2f})".format(dx, xmin, xmax))
print("dy: {:0.2f} (from {:.2f} to {:.2f})".format(dy, ymin, ymax))
print("dz: {:0.2f} (from {:.2f} to {:.2f})".format(dz, zmin, zmax))
print("box com: {}".format(box_com))

xx, yy, zz = np.meshgrid(xvals[:-1], yvals[:-1], zvals[:-1], indexing='ij')
# Center point of each voxel of V
gridpts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T + 0.5*np.array([dx,dy,dz])
n_voxels = gridpts.shape[0]

tree_grid = cKDTree(gridpts)

#cutoff = 7.0
cutoff = 7.0
sigma = 2.4

univ = MDAnalysis.Universe(args.top, args.traj)
start_frame = int(args.start / univ.trajectory.dt)
n_frames = univ.trajectory.n_frames


#if args.print_out:
#if True:
W = MDAnalysis.Writer("shift.xtc", univ.atoms.n_atoms)


futures = []


def task_gen(W=None, shifts=None):

    for i, i_frame, in enumerate(np.arange(start_frame, n_frames)):
        if i_frame % 100 == 0:
            print("frame: {}".format(i_frame))
            sys.stdout.flush()

        univ.trajectory[i_frame]
        
        ## Calculate cavity com pre-shit
        #if shift is not None:
        this_pos = univ.atoms.positions

        # Perform shift so cavity COM is centered in box
        if shifts is not None:
            shift_pos = this_pos + shifts[i]
            pbc(shift_pos, univ.dimensions[:3])
            univ.atoms.positions = shift_pos
        
            W.write(univ.atoms)
            if i_frame == n_frames - 1:
                univ.atoms.write("shift.gro")

        ## Find waters in V at this step
        waters = univ.select_atoms("name OW")
        water_pos = waters.positions

        selx = (water_pos[:,0] >= xmin-cutoff) & (water_pos[:,0] < xmax+cutoff)
        sely = (water_pos[:,1] >= ymin-cutoff) & (water_pos[:,1] < ymax+cutoff)
        selz = (water_pos[:,2] >= zmin-cutoff) & (water_pos[:,2] < zmax+cutoff)

        sel_mask = selx & sely & selz

        # Waters in V at this frame
        this_waters = waters[sel_mask]

        this_sam = univ.select_atoms("resname CH3 and not name H*")
        this_sam = this_sam[this_sam.positions[:,0] >= xmin-cutoff]

        fn_args = (this_waters.positions, this_sam.positions, tree_grid, nx, ny, nz)
        fn_kwargs = {'cutoff': cutoff, 'sigma': sigma, 'this_idx': i}
        

        yield (get_rhoxyz, fn_args, fn_kwargs)


## Calculate the cavity COM for each frame
cav_com = np.zeros((n_frames-start_frame, 3))
with wm:
    for future in wm.submit_as_completed(task_gen(), queue_size=wm.n_workers):
        idx, this_rho_water, this_rho_sam = future.get_result(discard=True)
        if idx % 10 == 0:
            print("getting result {}".format(idx))
            sys.stdout.flush()

        this_rho = ((this_rho_water/expt_waters) + (this_rho_sam/expt_sam))

        # cg cavity field (binary) at this frame
        this_cav = this_rho < 0.5
        cav_mask = this_cav.ravel()

        ## Large enough cavity - calculate its COM
        if cav_mask.sum()/gridpts.shape[0] > 1e-4:
            this_com = gridpts[cav_mask].mean(axis=0)
        else:
            this_com = box_com

        cav_com[idx, ...] = this_com

        #all_cav[idx, ...] = this_cav

        del this_rho_water, this_rho_sam, this_rho


## Now redo, but shift each frame so cavity com is centered in the box
all_cav = np.zeros((n_frames-start_frame, nx, ny, nz), dtype=np.bool)

print("Shifting each frame...")

shift_cav_com = box_com - cav_com
shift_cav_com[:,0] = 0

new_cav_com = np.zeros_like(cav_com)
max_verts = 0
all_verts = []
with wm:
    for future in wm.submit_as_completed(task_gen(W, shifts=shift_cav_com), queue_size=wm.n_workers):
        idx, this_rho_water, this_rho_sam = future.get_result(discard=True)
        if idx % 10 == 0:
            print("getting result {}".format(idx))
            sys.stdout.flush()

        this_rho = ((this_rho_water/expt_waters) + (this_rho_sam/expt_sam))

        # cg cavity field (binary) at this frame
        this_cav = this_rho < 0.5
        cav_mask = this_cav.ravel()

        if cav_mask.sum()/gridpts.shape[0] > 1e-4:
            this_com = gridpts[cav_mask].mean(axis=0)
        else:
            this_com = box_com

        new_cav_com[idx, ...] = this_com

        if cav_mask.sum() > 0:
            verts, faces, norms, vals = marching_cubes_lewiner(this_rho, level=0.5, spacing=np.array([dx, dy, dz]))
            if verts.shape[0] > max_verts:
                max_verts = verts.shape[0]

            verts = verts + np.array([xmin, ymin, zmin])   
        else:
            verts = np.array([])
        
        all_verts.append(verts)

        all_cav[idx, ...] = this_cav

        del this_rho_water, this_rho_sam, this_rho, this_cav

print("New cav COM: {}".format(new_cav_com.mean(axis=0)))
if not np.allclose(new_cav_com.mean(axis=0)[1:], box_com[1:], 0.01):
    print("Warning: shifted cavity com differs from box com...")


print("...Done: Outputting results")
np.savez_compressed("rhoxyz.dat", cav=all_cav, xbins=xvals, ybins=yvals, zbins=zvals, nframes=all_cav.shape[0], gridpts=gridpts, cav_com=shift_cav_com)
W.close()


#Finally, print out centered isosurface
univ_empty = MDAnalysis.Universe.empty(n_atoms=max_verts, trajectory=True)
W = MDAnalysis.Writer("inter.xtc", univ_empty.atoms.n_atoms)

for i, verts in enumerate(all_verts):

    univ_empty.atoms.positions = 0
    if verts.shape[0] > 0:
        
        this_n_verts = verts.shape[0]
        univ_empty.atoms[:this_n_verts].positions = verts
    if i == 0:
        univ_empty.atoms.write("inter.gro")

    W.write(univ_empty.atoms)

'''
### TEST
univ.trajectory[start_frame]
waters = univ.select_atoms("name OW")
water_pos = waters.positions

selx = (water_pos[:,0] >= xmin-cutoff) & (water_pos[:,0] < xmax+cutoff)
sely = (water_pos[:,1] >= ymin-cutoff) & (water_pos[:,1] < ymax+cutoff)
selz = (water_pos[:,2] >= zmin-cutoff) & (water_pos[:,2] < zmax+cutoff)

sel_mask = selx & sely & selz

# Waters in V at this frame
this_waters = waters[sel_mask]

this_sam = univ.select_atoms("resname CH3 and not name H*")
this_sam = this_sam[this_sam.positions[:,0] >= xmin-cutoff]

fn_args = (this_waters.positions, this_sam.positions, tree_grid, nx, ny, nz)
fn_kwargs = {'cutoff': cutoff, 'sigma': sigma, 'this_idx': 0}

i, this_rho_water, this_rho_sam = get_rhoxyz(*fn_args, **fn_kwargs)
this_rho = ((this_rho_water/expt_waters) + (this_rho_sam/expt_sam))
cav = ((this_rho_water/expt_waters) + (this_rho_sam/expt_sam)) < 0.5
cav_pts = gridpts[cav.ravel()]
univ.atoms.write("frame.gro")
this_waters.write("water_frame.gro")
this_sam.write("sam_frame.gro")
univ_empty = MDAnalysis.Universe.empty(n_atoms=cav_pts.shape[0], trajectory=True)
univ_empty.atoms.positions = cav_pts
univ_empty.atoms.write("cav.gro")

'''