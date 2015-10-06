#Instantaneous interface
# nrego sept 2015

import numpy
import argparse
import logging

import MDAnalysis
from MDAnalysis.coordinates.xdrfile.libxdrfile2 import read_xtc_natoms, xdrfile_open

import scipy.spatial
from skimage import measure

from utils import phi



if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Instantaneous interface analysis code. Either generate isosurface or analyze results")

    parser.add_argument('-c', '--grofile', metavar='INPUT', type=str, required=True,
                        help='Input structure file')
    parser.add_argument('-f', '--trajfile', metavar='XTC', type=str, required=True,
                        help='Input XTC trajectory file')
    parser.add_argument('-d', '--cutoff', type=float, default=7.0,
                        help='Cutoff distance for coarse grained gaussian, in Angstroms (default: 7 A)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='First timepoint (in ps)')
    parser.add_argument('-o', '--outfile', default='interface.dx',
                        help='Output file to write instantaneous interface')


    log = logging.getLogger('interface')


    args = parser.parse_args()


    cutoff = args.cutoff
    cutoff_sq = cutoff**2


    u = MDAnalysis.Universe(args.grofile, args.trajfile)

    # Start frame, in trajectory units
    startframe = int(args.start / u.trajectory.dt)
    lastframe = int(u.trajectory.n_frames / u.trajectory.dt)
    nframes = lastframe - startframe

    prot_heavies = u.select_atoms("not (name H* or resname SOL) and not (name CL or name NA)")
    water_ow = u.select_atoms("name OW")
    ions = u.select_atoms("name CL or name NA")

    # Hard coded for now - obviously must predict
    rho_water_bulk = 0.0330
    rho_prot_bulk = 0.040
    sigma = 2.4
    sigma_sq = sigma**2

    # uh...?
    grid_dl = 1
    natoms = u.coord.n_atoms

    # Numpy 6d array of xyz dimensions - I assume last three dimensions are axis angles?
    # should modify in future to accomodate non cubic box
    #   In angstroms
    box = u.dimensions


    # Set up marching cube stuff - grids in Angstroms
    #  ngrids are grid dimensions of discretized space at resolution ngrids[i] in each dimension
    ngrids = box[:3].astype(int)+1
    dgrid = box[:3]/(ngrids-1)

    # Extra number of grid points on each side to reflect pbc
    #    In grid units (i.e. inverse dgrid)
    #margin = (cutoff/dgrid).astype(int)
    margin = numpy.array([0,0,0], dtype=numpy.int)
    # Number of actual unique points
    npts = ngrids.prod()
    # Number of real points plus margin of reflected pts
    n_pseudo_pts = (ngrids + 2*margin).prod()

    # Construct 'gridpts' array, over which we will perform
    #    nearest neighbor searching for each heavy prot and water atom
    #  NOTE: gridpts is an augmented array - that is, it includes
    #  (margin+2) array points in each dimension - this is to reflect pbc
    #  conditions - i.e. atoms near the edge of the box can 'see' grid points
    #    within a distance (opp edge +- cutoff)
    #  I use an index mapping array to (a many-to-one mapping of gridpoint to actual box point)
    #     to retrieve appropriate real point indices
    coord_x = numpy.arange(-margin[0],ngrids[0]+margin[0],dgrid[0])
    coord_y = numpy.arange(-margin[1],ngrids[1]+margin[1],dgrid[1])
    coord_z = numpy.arange(-margin[2],ngrids[2]+margin[2],dgrid[2])
    xpts, ypts, zpts = numpy.meshgrid(coord_x,coord_y,coord_z)

    # gridpts array shape: (n_pseudo_pts, 3)
    #   gridpts npseudo unique points - i.e. all points
    #      on an enlarged grid
    gridpts = numpy.array(zip(xpts.ravel(),ypts.ravel(),zpts.ravel()))
    grididx = gridpts / dgrid
    # Many-to-one mapping of pseudo pt inx => actual point idx
    #   e.g. to translate from idx of gridpts array to point in
    #      'rho' arrays, below

    # This part looks like a fucking mess - try to clean it up to simplify a bit

    real_coord_x = coord_x.copy()
    real_coord_y = coord_y.copy()
    real_coord_z = coord_z.copy()

    real_coord_x[:margin[0]] += ngrids[0]
    real_coord_x[-margin[0]:] -= ngrids[0]

    real_coord_y[:margin[1]] += ngrids[1]
    real_coord_y[-margin[1]:] -= ngrids[1]

    real_coord_z[:margin[2]] += ngrids[2]
    real_coord_z[-margin[2]:] -= ngrids[2]

    x_real, y_real, z_real = numpy.meshgrid(real_coord_x, real_coord_y, real_coord_z)

    # Includes repeated points, all in units according to 'actual_pts' - shape: (n_pseudo_pts, 3)
    repeat_pts = (numpy.array(zip(x_real.ravel(), y_real.ravel(), z_real.ravel())) / dgrid).astype(int)


    pt_map = numpy.zeros((n_pseudo_pts,), dtype=numpy.int32)
    grid_sq = ngrids[0]**2
    grid = ngrids[1]
    for i in xrange(n_pseudo_pts):
        x,y,z = repeat_pts[i]
        idx = x*grid_sq + y*grid + z
        pt_map[i] = idx

    rho_water = numpy.zeros((nframes, npts), dtype=numpy.float32)
    rho_prot = numpy.zeros((nframes, npts), dtype=numpy.float32)
    rho = numpy.zeros((nframes, npts), dtype=numpy.float32)

    # KD tree for nearest neighbor search
    tree = scipy.spatial.cKDTree(gridpts)

    for i in xrange(startframe, lastframe):
        print "Frame: {}".format(i)

        u.trajectory[i]

        for atom in prot_heavies:

            pos = atom.position
            # Indices of all gridpoints within cutoff of atom's position
            neighboridx = numpy.array(tree.query_ball_point(pos, cutoff))
            real_idx = pt_map[neighboridx]
            neighborpts = gridpts[neighboridx]

            dist_vectors = neighborpts[:, ...] - pos

            # Distance array between atom and neighbor grid points
            #distarr = scipy.spatial.distance.cdist(pos.reshape(1,3), neighborpts,
            #                                       'sqeuclidean').reshape(neighboridx.shape)

            phivals = phi(dist_vectors, sigma, sigma_sq, cutoff, cutoff_sq)

            rho_prot[i-startframe, real_idx] += phivals

        for atom in water_ow:
            pos = atom.position
            neighboridx = numpy.array(tree.query_ball_point(pos, cutoff))
            real_idx = pt_map[neighboridx]
            neighborpts = gridpts[neighboridx]

            dist_vectors = neighborpts[:, ...] - pos
            # Distance array between atom and neighbor grid points
            # distarr = scipy.spatial.distance.cdist(pos.reshape(1,3),
            #       neighborpts,'sqeuclidean').reshape(neighboridx.shape)

            phivals = phi(dist_vectors, sigma, sigma_sq, cutoff, cutoff_sq)

            rho_water[i-startframe, real_idx] += phivals

        rho[i-startframe, :] = rho_prot[i-startframe, :]/rho_prot_bulk \
            + rho_water[i-startframe, :]/rho_water_bulk
        # rho[i-startframe, :] /= 2.0

    # Hack out the last frame to a volumetric '.dx' format (readable by VMD)
    prot_tree = scipy.spatial.cKDTree(prot_heavies.positions)
    rho_shape = (rho.sum(axis=0)/rho.shape[0]).reshape(ngrids)
    #rho_shape = rho[0].reshape(ngrids)
    outfile = args.outfile
    cntr = 0
    with open(outfile, 'w') as f:
        f.write("object 1 class gridpositions counts {} {} {}\n".format(ngrids[0], ngrids[1], ngrids[2]))
        f.write("origin {:1.8e} {:1.8e} {:1.8e}\n".format(0,0,0))
        f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(dgrid[0], 0, 0))
        f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0, dgrid[1], 0))
        f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0, 0, dgrid[2]))
        f.write("object 2 class gridconnections counts {} {} {}\n".format(ngrids[0], ngrids[1], ngrids[2]))
        f.write("object 3 class array type double rank 0 items {} data follows\n".format(npts))
        for i in xrange(ngrids[0]):
            for j in xrange(ngrids[1]):
                for k in xrange(ngrids[2]):
                    grid_pt = numpy.array([dgrid[0]*i, dgrid[1]*j, dgrid[2]*k])
                    dist, idx = prot_tree.query(grid_pt, distance_upper_bound=10.0)

                    if dist == float('inf'):
                        rho_shape[i,j,k] += 1.0
                    #if (i < lowercutoff_x or j < lowercutoff_y or k < lowercutoff_z) or (i > uppercutoff_x or j > uppercutoff_y or k > uppercutoff_z):
                    #    rho_shape[i,j,k] += 1.0
                    f.write("{:1.8e} ".format(rho_shape[i,j,k]))
                    cntr += 1
                    if (cntr % 3 == 0):
                        f.write("\n")
