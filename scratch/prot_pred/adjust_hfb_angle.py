from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import MDAnalysis

from IPython import embed


# Find angle from hfb com to hfb's hydrophobic patch com, align with z axis

univ = MDAnalysis.Universe('prot.gro')

prot_ag = univ.select_atoms('protein and not name H*')
phob_ag = univ.select_atoms('resid 7 or resid 18 or resid 19 or resid 21 or resid 22 ' 
                            'or resid 24 or resid 54 or resid 55 or resid 57 '
                            'or resid 58 or resid 61 or resid 62 or resid 63')
phob_ag = phob_ag.select_atoms('not name H*')

# move protein cog to origin

univ.atoms.positions -= prot_ag.center_of_geometry()
phob_cog = phob_ag.center_of_geometry().astype(float)

# Normalized vector from origin to cog of hydrophobic patch atoms
diff_vec = phob_cog / np.linalg.norm(phob_cog)

z_vec = np.array([0.,0.,1.])

dot_c = np.dot(diff_vec, z_vec)
angle = np.arccos(dot_c)

axis_vec = np.cross(diff_vec, z_vec)
axis_vec = axis_vec / np.linalg.norm(axis_vec)
s = np.linalg.norm(axis_vec)

K = np.array([[0, -axis_vec[2], axis_vec[1]], 
              [axis_vec[2], 0, -axis_vec[0]], 
              [-axis_vec[1], axis_vec[0], 0]])

# Rotation matrix R
R = np.eye(3) + np.sin(angle)*K + (1-dot_c)*K.dot(K)

univ.atoms.positions = univ.atoms.positions.dot(R.T)
univ.atoms.write('aligned.pdb')