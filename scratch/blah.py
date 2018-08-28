from __future__ import division

import numpy as np
import MDAnalysis


ref_univ = MDAnalysis.Universe('equil.gro')
ref_univ.add_TopologyAttr('tempfactor')
ag = ref_univ.select_atoms('name S1')


z_space = 5.0 # 5 A
y_space = np.sqrt(3.0)/2.0 * z_space
curr_y = -y_space
curr_z = 0

nz = 12
ny = nz#+2
zlen = nz * z_space
ylen = ny * y_space
npts = nz * ny
natoms = 13 * npts

ag = ag[:npts]

gridpts = np.zeros((npts, 3), dtype=float)

row_num = -1
indices = []
not_seen = np.ones(npts, dtype=bool)
for i in range(npts):

    # Go to new row 
    if i % nz == 0:
        row_num += 1
        curr_y += y_space
        if (i/nz) % 2 == 0:
            curr_z = 0.0
        else:
            curr_z = - z_space/2.0
    else:
        curr_z += z_space

    # check if we're a nm away from edges
    cent = row_num > 2 and row_num < ny-3 and i % nz > 2 and i % nz < nz - 3

    gridpts[i] = 0, curr_y, curr_z

    this_pt = gridpts[i]
    min_idx = i

    min_atm = ag[not_seen][min_idx]
    min_res = ref_univ.residues[min_atm.resid-1]

    shift_vec = this_pt - min_atm.position

    for atm in min_res.atoms:
        atm.position += shift_vec
        indices.append(atm.index)
        if cent:
            atm.tempfactor = 0
        else:
            atm.tempfactor = 1

ag = ref_univ.atoms[indices]
ag.atoms.write('small.pdb')
