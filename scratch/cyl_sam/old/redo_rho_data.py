
import numpy as np

import argparse
import logging

import scipy

import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed
import math

import sys

from whamutils import get_negloghist, extract_and_reweight_data

import argparse



fnames = sorted(glob.glob('*/rhoxyz.dat.npz'))

ds = np.load(fnames[0])

xbins = ds['xbins']
ybins = ds['ybins']
zbins = ds['zbins']
gridpts = ds['gridpts']

dx = np.diff(xbins)[0]
dy = np.diff(ybins)[0]
dz = np.diff(zbins)[0]

expt_waters = 0.033 * dx * dy * dz

for fname in fnames:
    print("Doing {}".format(fname))
    dirname = os.path.dirname(fname)
    ds = np.load(fname)

    rho = ds['rho']

    cav = (rho / expt_waters) < 0.5

    assert np.array_equal(ds['xbins'], xbins)
    assert np.array_equal(ds['ybins'], ybins)
    assert np.array_equal(ds['zbins'], zbins)
    assert np.array_equal(ds['gridpts'], gridpts)
    ds.close()
    del rho

    np.savez_compressed("{}/cg_rho.dat".format(dirname), cav=cav, xbins=xbins, ybins=ybins, zbins=zbins, gridpts=gridpts)


