from __future__ import division, print_function

import numpy as np
import matplotlib
mpl = matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage, imread
from scipy.optimize import minimize


#import visvis as vv

#from IPython import embed

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from mdtools import dr

from constants import k

import os
import glob

from mdtools import dr

beta = 1/(k*300)


payload = np.load('boot_fn_payload.dat.npy')

n_boot_sample = payload.shape[0]
binbounds = np.arange(-180,184,4)

loghists = []

for i in range(n_boot_sample):
    weights, boot_data = payload[i]

    # Get everything with phi < 0
    mask = boot_data[:,0] < 0
    hist, bb = np.histogram(boot_data[mask,1], bins=binbounds, weights=weights[mask])

    loghist = -np.log(hist)
    loghist -= loghist.min()

    loghists.append(loghist)

loghists = np.array(loghists)

avg = loghists.mean(axis=0)
errs = loghists.std(axis=0, ddof=1)

plt.errorbar(binbounds[:-1], avg, yerr=errs)