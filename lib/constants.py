#TODO: Move this general selection spec somewhere globally accessible
#   It selects all atoms that are NOT:
#   
#   Hydrogen (name H*)
#   Solvent (molname SOL)
#   wall (resname WAL)
#   ions (CL and NA only)
#   dummy (name DUM)
SEL_SPEC_HEAVIES_NOWALL = "not (name H* or resname SOL or resname WAL) and not (name CL or name NA or name DUM)"
SEL_SPEC_NOWALL = "not (resname SOL or resname WAL) and not (name CL or name NA or name DUM)"

SEL_SPEC_HEAVIES = "not (name H* or resname SOL) and not (name CL or name NA or name DUM) and not resname INT"
SEL_SPEC = "not resname SOL and not (name CL or name NA or name DUM) and not resname INT"

SEL_SPEC_NOT_HEAVIES_NOWALL = "not (not (name H* or resname SOL or resname WAL) and not (name CL or name NA or name DUM))"
SEL_SPEC_NOT_NOWALL = "not (not (resname SOL or resname WAL) and not (name CL or name NA or name DUM))"

# boltzmann constant in kJ/(mol*K)
k = 8.3144598e-3