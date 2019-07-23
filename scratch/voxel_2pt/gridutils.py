import numpy as np
import MDAnalysis

from scipy.spatial import cKDTree
import itertools

# Will construct nx*ny*nz gridpoints, where nx=x_bounds.size-1, etc
def construct_gridpts(x_bounds, y_bounds, z_bounds):
    dgrid = np.diff(x_bounds)[0]
    #assert np.unique(np.diff(x_bounds)).size == 1

    xx, yy, zz = np.meshgrid(x_bounds[:-1], y_bounds[:-1], z_bounds[:-1], indexing='ij')
    gridpts = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T + 0.5*dgrid

    return gridpts


## Save a list of gridpoints to a pdb file. Optionally color by list of tempfactors
def save_gridpts(fout, gridpts, tempfactors=None):

    n_pts = gridpts.shape[0]
    univ = MDAnalysis.Universe.empty(n_pts, n_pts, atom_resindex=np.arange(n_pts), trajectory=True)
    univ.add_TopologyAttr('name')
    univ.add_TopologyAttr('resname')
    univ.add_TopologyAttr('id')
    univ.add_TopologyAttr('tempfactors')

    univ.residues.resnames = 'V'
    univ.atoms.names = 'V'
    univ.atoms.positions = gridpts

    if tempfactors is not None:
        assert tempfactors.size == gridpts.shape[0]
        univ.atoms.tempfactors = tempfactors


    univ.atoms.write(fout)
