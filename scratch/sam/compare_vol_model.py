from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

### Try to construct linear model for pure surfaces, comparing volumes in bulk (for possible correction for edge effects)

ds_bulk = np.load('sam_pattern_bulk_pure.npz')
ds_pure = np.load('sam_pattern_pure.npz')