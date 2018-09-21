from __future__ import division; __metaclass__ = type
import sys
import numpy as np
from math import sqrt
import argparse
import logging

import MDAnalysis

def norm(vec):
    return np.sqrt( np.sum(vec**2) )

infile = sys.argv[1]

univ = MDAnalysis.Universe(infile)

patch = univ.residues[-36:].atoms

head_groups = patch.select_atoms('name CT or name O12')
ch3 = patch.select_atoms('name CT')

head_cog = head_groups.center_of_geometry()
patch_cog = ch3.center_of_geometry()

d = norm(head_cog-patch_cog)
rms = np.sqrt( np.mean(np.sum((ch3.positions - patch_cog)**2, axis=1) ) )


print("rms: {}   dist from cent: {}".format(rms, d))