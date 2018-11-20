import numpy as np

import MDAnalysis

univ = MDAnalysis.Universe('struct.gro')

c0 = univ.residues[0].atoms.select_atoms('name C')[0].position.astype(np.float64)
n0 = univ.residues[1].atoms.select_atoms('name N')[0].position.astype(np.float64)
c1 = univ.residues[1].atoms.select_atoms('name C')[0].position.astype(np.float64)
n1 = univ.residues[2].atoms.select_atoms('name N')[0].position.astype(np.float64)
ca = univ.residues[1].atoms.select_atoms('name CA')[0].position.astype(np.float64)

p0 = np.cross( (c0-n0), (ca-n0))
p0 /= np.linalg.norm(p0)

p1 = np.cross((n0-ca), (c1-ca))
p1 /= np.linalg.norm(p1)

