from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import os, glob, shutil

from IPython import embed

import logging
logging.captureWarnings(True)

fnames = sorted(glob.glob('*/prot_heavies_by_charge.pdb'))

for fname in fnames:

    dirname = os.path.dirname(fname)
    print('doing {}'.format(dirname))

    u1 = MDAnalysis.Universe(fname)
    u2 = MDAnalysis.Universe('{}/cluster_h.pdb'.format(dirname))
    cog = u1.atoms.center_of_geometry()

    u1.atoms.positions -= cog
    u2.atoms.positions -= cog

    u1.atoms.write('{}/{}_struct_h.pdb'.format(dirname, dirname))
    u2.atoms.write('{}/{}_clust_h.pdb'.format(dirname, dirname))