from __future__ import division

import numpy as np
import MDAnalysis

import argparse
from IPython import embed
import os, glob, pathlib

import math

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scratch.sam.util import *

beta = 1 / (k*300)
kappa = beta*0.386

def extract_pq(name):
    splits = name.split('_')

    return int(splits[1]), int(splits[3])


## Collect all *new* uniform OH- surface free energies

ds_bulk = np.load("sam_pattern_bulk_pure.npz")
ds_old = np.load("old_sam_pattern_pure.npz")

e_bulk = ds_bulk['energies']
pq = ds_bulk['pq']
e_old = ds_old['energies'][1::2]

assert np.array_equal(ds_old['pq'][1::2], pq)

dnames = glob.glob('P_*/k_00/d_*/trial_0')

e_new = np.zeros_like(e_old)
err_new = np.zeros_like(e_new)

for dname in dnames:
    path = pathlib.Path(dname)
    p,q = extract_pq(path.parts[0])

    idx = np.where((pq[:,0] == p) & (pq[:,1] == q))[0].item()

    test_p, test_q = pq[idx]
    assert test_p == p
    assert test_q == q

    print("doing p : {} q : {}".format(p,q))

    pvn = np.loadtxt("{}/PvN.dat".format(dname))

    e_new[idx] = pvn[0,1]
    err_new[idx] = pvn[0,2]


tot_new_energies = np.zeros_like(ds_old['energies'])
tot_new_energies[::2] = ds_old['energies'][::2]
tot_new_energies[1::2] = e_new

tot_new_errs = np.zeros_like(ds_old['err_energies'])
tot_new_errs[::2] = ds_old['err_energies'][::2]
tot_new_errs[1::2] = err_new

np.savez_compressed('sam_pattern_pure', energies=tot_new_energies, pq=ds_old['pq'], err_energies=tot_new_errs, 
                    dx=ds_old['dx'], dy=ds_old['dy'], dz=ds_old['dz'], base_energy=ds_old['base_energy'], base_err=ds_old['base_err'])


