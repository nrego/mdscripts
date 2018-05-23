#TODO: Move this general selection spec somewhere globally accessible
#   It selects all atoms that are NOT:
#   
#   Hydrogen (name H*)
#   Solvent (molname SOL)
#   wall (resname WAL)
#   ions (CL and NA only)
#   dummy (name DUM)
sel_spec_heavies_nowall = "not (name H* or resname SOL or resname WAL) and not (name CL or name NA or name DUM)"
sel_spec_nowall = "not (resname SOL or resname WAL) and not (name CL or name NA or name DUM)"

sel_spec_heavies = "not (name H* or resname SOL) and not (name CL or name NA or name DUM) and not resname INT"
sel_spec = "not resname SOL and not (name CL or name NA or name DUM) and not resname INT"

sel_spec_not_heavies_nowall = "not (not (name H* or resname SOL or resname WAL) and not (name CL or name NA or name DUM))"
sel_spec_not_nowall = "not (not (resname SOL or resname WAL) and not (name CL or name NA or name DUM))"