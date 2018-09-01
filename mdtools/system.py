from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import MDAnalysis
from mdtools import dr

from IPython import embed

import logging
log = logging.getLogger(__name__)

logging.captureWarnings(True)


BURIED = -2
HYDROPHILIC = -1
HYDROPHOBIC = 0

class MDSystem():
    """
    Abstraction for MD systems

    For analysis on equilibrium systems

    rho_ref: shape (n_prot_heavies, ): average number of waters per protein
        heavy atom
    """
    def __init__(self, top, struct, rho_ref, **kwargs):

        self.univ = MDAnalysis.Universe(top, struct)
        self.univ.add_TopologyAttr('tempfactor')
        self.prot = self.univ.select_atoms('protein')
        self.prot_h = self.univ.select_atoms('protein and not name H*')
        self.prot.tempfactors = 0

        self.rho_ref = rho_ref
        assert rho_ref.size == self.prot_h.n_atoms

        for k,v in kwargs.iteritems():
            self.k = v

    @property
    def n_prot_tot(self):
        return self.prot.n_atoms

    @property
    def n_prot_h_tot(self):
        return self.prot_h.n_atoms
    
    @property
    def n_surf(self):
        return (self.prot.tempfactors != BURIED).sum()

    @property
    def n_surf_h(self):
        return (self.prot_h.tempfactors != BURIED).sum()

    @property
    def n_buried(self):
        return (self.prot.tempfactors == BURIED).sum()

    @property
    def n_buried_h(self):
        return (self.prot_h.tempfactors == BURIED).sum()
    
    @property
    def n_phil(self):
        phil = self.prot.tempfactors == HYDROPHILIC
        surf = self.prot.tempfactors != BURIED

        return (phil & surf).sum()

    @property
    def n_phob(self):
        phob = self.prot.tempfactors == HYDROPHOBIC
        surf = self.prot.tempfactors != BURIED

        return (phob & surf).sum()

    @property
    def n_phil_h(self):
        phil = self.prot_h.tempfactors == HYDROPHILIC
        surf = self.prot_h.tempfactors != BURIED

        return (phil & surf).sum()

    @property
    def n_phob_h(self):
        phob = self.prot_h.tempfactors == HYDROPHOBIC
        surf = self.prot_h.tempfactors != BURIED

        return (phob & surf).sum()

    def find_buried(self, nb):
        buried_mask = self.rho_ref < nb
        self.prot_h[buried_mask].tempfactors = -2

        for atm in self.prot:
            # If hydrogen, check if the atom it's bound to is buried
            if atm.name[0] == 'H':  
                atm.tempfactor = atm.bonded_atoms[0].tempfactor

    def assign_hydropathy(self, charge_assign):

        for atm in self.prot:
            # skip if buried
            if atm.tempfactor == BURIED:
                continue
            try:
                hydrophil = charge_assign[atm.resname][atm.name]
            except:
                hydrophil = float(raw_input('enter hydrophilicity for atom {} of {} (-1 for hydrophilic, 0 for hydrophobic):  '.format(atm.name, atm.resname)))
                charge_assign[atm.resname][atm.name] = hydrophil
            atm.tempfactor = hydrophil



