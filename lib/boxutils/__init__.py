import numpy as np
from MDAnalysis import SelectionError
from MDAnalysis.analysis.align import alignto
from utils import pbc

from selection_specs import sel_spec_heavies_nowall

# centers 'mol' (specified by mdanalysis selection string, 'mol_spec') in box
#   Translates all atoms in 'univ' so COM of mol is in the center of box. Original
#     orientations maintained
#
#  Modifies MDAnalysis Universe 'univ' in-place by working on atom groups
def center_mol(univ, mol_spec=sel_spec_heavies_nowall):


    return univ


# rotates (an already centered!) system to minimize the rotational RMSD
#   of 'mol' against a reference structure. Applies rotation to all
#   atoms in system
#
#  Modifies MDAnalysis Universe 'univ' in-place
def rotate_mol(ref_univ, univ_other, mol_spec=sel_spec_heavies_nowall):

    alignto(univ_other, ref_univ, select=mol_spec)

    return ref_univ