
outer_ring_idx = np.array([1,8,14,19,20,21,22,18,13,7,28,33,37,36,35,34,29,23])

center_ring_idx_targ = np.array([3,10,25,5,11,26])
center_ring_idx_part = np.array([5,11,26,3,10,25])

center_pt = 3

pattern = np.zeros(37, dtype=int)
np.savetxt('pattern_phobic.dat', pattern, fmt='%d')


univ = MDAnalysis.Universe('top.tpr', 'equil.gro')

targ = univ.select_atoms("segid seg_0_Plate")
part = univ.select_atoms("segid seg_1_Plate_part")

# make outer ring hydrophilic
res_targ = targ.residues[outer_ring_idx-1]
for res in res_targ:
    atm = res.atoms[0]
    atm.name = 'OH'
    res.resname = 'OH'
try:
    res_part = part.residues[outer_ring_idx-1]
    for res in res_part:
        atm = res.atoms[0]
        atm.name = 'OH'
        res.resname = 'OH'
except IndexError:
    pass

pattern[outer_ring_idx-1] = 1
np.savetxt('pattern_polar_ring.dat', pattern, fmt='%d')
univ.atoms.write("polar_ring.gro")

# Make charged center
res_targ = targ.residues[center_ring_idx_targ-1]
for i,res in enumerate(res_targ):
    newtype = 'P' if i < 3 else 'N'
    atm = res.atoms[0]
    atm.name = newtype
    res.resname = newtype
try:
    res_part = part.residues[center_ring_idx_part-1]
    for i,res in enumerate(res_part):

        newtype = 'N' if i < 3 else 'P'

        atm = res.atoms[0]
        atm.name = newtype
        res.resname = newtype
except IndexError:
    pass

atm = targ.residues[3].atoms[0]
atm.name = 'OH'
atm.residue.resname = 'OH'
try:
    atm = part.residues[3].atoms[0]
    atm.name = 'OH'
    atm.residue.resname = 'OH'
except IndexError:
    pass

for i, idx in enumerate(center_ring_idx_targ):

    newtype = 3 if i < 3 else 2

    pattern[idx-1] = newtype

pattern[3] = 1
np.savetxt('pattern_charged.dat', pattern, fmt='%d')
univ.atoms.write("charged.gro")
