from __future__ import division, print_function; __metaclass__ = type
import numpy as np
import re
import MDAnalysis

from rhoutils import cartesian

import pandas
import mdtraj as md

from skimage import measure
#from IPython import embed

def extractInt(string):
    return list(map(int, re.findall(r"[-+]?\d*\.\d+|\d+", string)))

# Attempt to intialize from .dx file
def from_dx(infile):

    with open(infile, 'r') as fin:

        # ngrids
        lines = fin.readlines()

        origin = np.array(lines[2].split()[1:]).astype(float)

        ngrids = np.zeros(3, dtype=int)
        dgrid = np.zeros(3)

        ngrids[0] = int(lines[3].split()[0])
        dgrid[0] = float(lines[3].split()[1])
        ngrids[1] = int(lines[4].split()[0])
        dgrid[1] = float(lines[4].split()[2])
        ngrids[2] = float(lines[5].split()[0])
        dgrid[2] = float(lines[5].split()[3])

        npts = ngrids.prod()

        box = ((ngrids) * dgrid) + origin + dgrid

        xpts = np.arange(origin[0], box[0], dgrid[0]) 
        ypts = np.arange(origin[1], box[1], dgrid[1])
        zpts = np.arange(origin[2], box[2], dgrid[2])

        xx, yy, zz = np.meshgrid(xpts[:-1], ypts[:-1], zpts[:-1], indexing='ij')
        gridpts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T + 0.5*dgrid

        rho = np.zeros((npts), dtype=np.float32)
        curr_idx = 0
        for line in lines[7:]:
            parts = line.split()
            rho[curr_idx:curr_idx+len(parts)] = parts
            curr_idx += len(parts)

        rho = rho.astype(float)
        rho_reshape = np.reshape(rho, (ngrids))
        rho_reshape = rho_reshape[np.newaxis, :]

    return rho, rho_reshape, gridpts


#Input/output capabilities for handling volumetric density fields on a grid
#  Includes GRO, XTC, and DX write capabilities
#
#  Initialization:
#    rho: either shape (n_grid_x, n_grid_y, n_grid_z) 
#                   or (n_frames, n_grid_x, n_grid_y, n_grid_z)
#    dt: (optional) time step between frames, if provided
class RhoField:

    # Allowed output extensions
    _output_ext = ['GRO', 'XTC', 'DX']
    def __init__(self, rho, gridpts, weights=None, time=None):

        self.rho = None
        self.rho_avg = None
        self.gridpts = None
        self.meshpts = None

        self._n_pts = None
        self._n_grids = None
        self._d_grid = None
        self._box = None

        if rho.ndim == 3:
            rho = rho[np.newaxis,...]

        assert rho.ndim == 4

        # Shape is (n_frames, n_grid_x, n_grid_y, n_grid_z)
        self.rho = rho

        assert gridpts.shape == (self.n_pts, 3)

        # Check that grid is uniformly spaced in each dimension
        assert np.unique(np.diff(np.unique(gridpts[:,0]))).size == 1
        assert np.unique(np.diff(np.unique(gridpts[:,1]))).size == 1
        assert np.unique(np.diff(np.unique(gridpts[:,1]))).size == 1

        self.gridpts = gridpts

        if time is None:
            time = np.arange(self.n_frames)
        self.time = time
        # Shape: (n_grid_x, n_grid_y, n_grid_z)
        self._get_rho_avg(weights)

        self._generate_mesh()


    @property 
    def n_grids(self):
        if self._n_grids is None:
            self._n_grids = np.array(self.rho.shape[1:])

        return self._n_grids

    @property 
    def n_pts(self):
        return self.n_grids.prod()

    @property
    def n_frames(self):
        return self.rho.shape[0]

    @property
    def d_grid(self):
        if self._d_grid is None:
            d_x = np.unique(np.diff(np.unique(self.gridpts[:,0])))[0]
            d_y = np.unique(np.diff(np.unique(self.gridpts[:,1])))[0]
            d_z = np.unique(np.diff(np.unique(self.gridpts[:,2])))[0]

            self._d_grid = np.array([d_x, d_y, d_z])

        return self._d_grid

    #returns (3,3) array of box vectors; currently only supports cubic box
    @property
    def box(self):
        if self._box is None:
            xvec,yvec,zvec = (self.d_grid*self.n_grids) - 1
            self._box = np.diag((xvec,yvec,zvec))

        return self._box
    
    
    def _get_rho_avg(self, weights=None):

        if weights is None:
            weights = np.ones(self.n_frames)
        
        assert weights.ndim == 1
        assert self.rho.shape[0] == weights.size

        weights /= weights.sum()

        self.rho_avg = np.sum((self.rho.T * weights).T, axis=0).reshape(self.n_grids)

        assert self.rho_avg.shape[0] == self.rho.shape[1]

    def _generate_mesh(self, isoval=0.5):

        # can't guarantee each frame will have the same number of mesh points
        meshpts = np.zeros((self.n_frames,), dtype=np.object)

        max_pts = 0
        for i_frame in range(self.n_frames):
            try:
                #embed()
                verts, faces, normals, values = measure.marching_cubes(self.rho[i_frame], isoval, spacing=tuple(self.d_grid))
                mesh = verts
            except ValueError:
                mesh = np.zeros((1,3))
            if mesh.shape[0] > max_pts:
                max_pts = mesh.shape[0]
            meshpts[i_frame] = mesh

        self.meshpts = np.zeros((self.n_frames, max_pts, 3))
        # artificially require each mesh to have the same number of points
        for i, mesh in enumerate(meshpts):
            # fill with dummy points
            dummy_pts = np.ones((max_pts-mesh.shape[0], 3))
            dummy_pts[:] = mesh[0]
            self.meshpts[i] = np.append(mesh, dummy_pts, axis=0)

        del meshpts

    # If 'frame' is specified, write out specific frame
    #   Otherwise, write out rho_avg
    #   'time', optional, is timepoint of current frame.
    def do_GRO(self, fileout, frame=None, top=None):
        #embed()
        if frame is not None:
            mesh = self.meshpts[frame]
            curr_time = np.array([self.time[frame]])
        else:
            mesh = self.meshpts[0]

        n_atoms = mesh.shape[0]
        mesh = mesh[np.newaxis, ...] 

        if top is None:
            top = md.Topology()
            c = top.add_chain()

            cnt = 0
            for i in range(n_atoms):
                cnt += 1
                r = top.add_residue('II', c)
                a = top.add_atom('II', md.element.get_by_symbol('VS'), r, i)

        with md.formats.PDBTrajectoryFile(fileout, 'w') as f:
            # Mesh pts have to be in nm
            f.write(mesh/10, top, unitcell_vectors=self.box.reshape(1,3,3)/10)

    # time : array shape (n_frames); sim time for each frame, in ps
    def do_XTC(self, fileout):
    
        with md.formats.XTCTrajectoryFile(fileout, 'w') as f:
            for i_frame in range(self.n_frames):
                curr_time = self.time[i_frame]
                mesh = self.meshpts[i_frame]
                mesh = mesh[np.newaxis, ...]

                f.write(mesh/10, time=curr_time, box=self.box/10)

    def do_DX(self, fileout, origin=(0,0,0)):
        cntr = 0

        rho_shape = self.rho_avg.reshape(self.n_pts)
        with open(fileout, 'w') as f:
            f.write("object 1 class gridpositions counts {} {} {}\n".format(self.n_grids[0], self.n_grids[1], self.n_grids[2]))
            f.write("origin {:1.8e} {:1.8e} {:1.8e}\n".format(*origin))
            f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(self.d_grid[0], 0, 0))
            f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0, self.d_grid[1], 0))
            f.write("delta {:1.8e} {:1.8e} {:1.8e}\n".format(0, 0, self.d_grid[2]))
            f.write("object 2 class gridconnections counts {} {} {}\n".format(self.n_grids[0], self.n_grids[1], self.n_grids[2]))
            f.write("object 3 class array type double rank 0 items {} data follows\n".format(self.n_pts))

            for pt_idx, grid_pt in enumerate(self.gridpts):

                f.write("{:1.8e} ".format(rho_shape[pt_idx]))
                cntr += 1
                if (cntr % 3 == 0):
                    f.write("\n")

