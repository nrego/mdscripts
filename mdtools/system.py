from __future__ import print_function, division; __metaclass__ = type

import numpy as np
import MDAnalysis
from mdtools.datareader import dr

from IPython import embed

import logging
log = logging.getLogger(__name__)

logging.captureWarnings(True)


BURIED = -2
HYDROPHILIC = -1
HYDROPHOBIC = 1
NULL = 0

class MDSystem():
    """
    Abstraction for MD systems

    For analysis on equilibrium systems

    rho_ref: shape (n_prot_heavies, ): average number of waters per protein
        heavy atom
    """
    def __init__(self, top, struct, sel_spec='segid targ', **kwargs):
        self.univ = MDAnalysis.Universe(top, struct)
        self.univ.add_TopologyAttr('tempfactor')
        for seg in self.univ.segments:
            seg.segid = seg.segid.split('_')[-1]

        self.prot = self.univ.select_atoms('({})'.format(sel_spec))
        self.prot_h = self.univ.select_atoms('({}) and not name H*'.format(sel_spec))
        self.hydrogens = self.univ.select_atoms('({}) and name H*'.format(sel_spec))
        self.prot.tempfactors = NULL
        self.other = self.univ.select_atoms('not ({})'.format(sel_spec))
        
        for k,v in kwargs.items():
            self.__dict__[k] = v
        

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
    def phobic_mask(self):
        return self.prot.tempfactors == HYDROPHOBIC

    @property
    def phobic_mask_h(self):
        return self.prot_h.tempfactors == HYDROPHOBIC

    @property
    def philic_mask(self):
        return self.prot.tempfactors == HYDROPHILIC

    @property
    def philic_mask_h(self):
        return self.prot_h.tempfactors == HYDROPHILIC

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

    @property
    def surf_mask(self):
        return self.prot.tempfactors != BURIED

    @property
    def surf_mask_h(self):
        return self.prot_h.tempfactors != BURIED
    
    

    # force all hydrogens to inherit tempfactors from heavy atom
    def _apply_to_h(self):
        for atm in self.hydrogens:
            atm.tempfactor = atm.bonded_atoms[0].tempfactor

    def find_buried(self, rho_dat, nb=5):
        
        self.prot[~self.surf_mask].tempfactors = NULL
        buried_mask = rho_dat < nb

        self.prot_h[buried_mask].tempfactors = BURIED

        self._apply_to_h()

    def assign_hydropathy(self, charge_assign):

        for atm in self.prot:
            # skip if buried
            if atm.tempfactor == BURIED:
                continue
            try:
                hydrophil = charge_assign[atm.resname][atm.name]
            except:
                hydrophil = float(input('enter hydrophilicity for atom {} of {} (-1 for polar/charged, 1 for non-polar):  '.format(atm.name, atm.resname)))
                try: 
                    charge_assign[atm.resname][atm.name] = hydrophil
                except KeyError:
                    charge_assign[atm.resname] = dict()
                    charge_assign[atm.resname][atm.name] = hydrophil

            atm.tempfactor = hydrophil




