#TODO: Move this general selection spec somewhere globally accessible
#   It selects all atoms that are NOT:
#   
#   Hydrogen (name H*)
#   Solvent (molname SOL)
#   wall (resname WAL)
#   ions (CL and NA only)
#   dummy (name DUM)
sel_spec_heavies_nowall = "not (name H* or resname SOL or resname WAL) and not (name CL or name NA or name DUM)"