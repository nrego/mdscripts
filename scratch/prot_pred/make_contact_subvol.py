import MDAnalysis
import numpy as np

dat = np.loadtxt("actual_contact_mask.dat", dtype=bool)

univ = MDAnalysis.Universe("../equil/equil.tpr", "../equil/cent.gro")
prot = univ.select_atoms("(segid seg_0_Protein_chain_A or segid seg_1_Protein_chain_B)and not name H*")

assert dat.size == prot.n_atoms
contact_atoms = prot[dat]

header_string = "; Umbrella potential for a spherical shell cavity\n"\
"; Name    Type          Group  Kappa   Nstar    mu    width  cutoff  outfile    nstout\n"\
"hydshell dyn_union_sph_sh   OW  0.0     0   XXX    0.01   0.02   phiout.dat   50  \\\n"

with open('umbr_contact.conf', 'w') as fout:
    fout.write(header_string)

    for atm in contact_atoms:
        fout.write("{:<10.1f} {:<10.1f} {:d} \\\n".format(-0.5, 0.6, atm.index+1))

        