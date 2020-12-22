
import sys
import numpy as np
from math import sqrt
import argparse
import logging
import shutil
import MDAnalysis

from IPython import embed

from scipy.spatial import cKDTree
import itertools
#from skimage import measure

from rhoutils import rho, cartesian
from mdtools import ParallelTool

from constants import SEL_SPEC_HEAVIES, SEL_SPEC_HEAVIES_NOWALL
from mdtools.fieldwriter import RhoField
import sys
import argparse, os
from scipy.interpolate import interp2d

import matplotlib as mpl

from skimage import measure
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

myorange = myorange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1] 


homedir = os.environ['HOME']


mpl.rcParams.update({'axes.labelsize': 50})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize': 30})

dat_03 = np.loadtxt("w_03/data_reduced/theta_v_n.dat") 
dat_09 = np.loadtxt("w_09/data_reduced/theta_v_n.dat")

mse_03 = np.loadtxt("w_03/data_reduced/mse_v_n.dat")
mse_09 = np.loadtxt("w_09/data_reduced/mse_v_n.dat")

expt_theta = 49.5

nvphi_03 = np.loadtxt("w_03/NvPhi.dat")
nvphi_09 = np.loadtxt("w_09/NvPhi.dat")

max_idx_03 = np.argmax(nvphi_03[:,2])
max_idx_09 = np.argmax(nvphi_09[:,2])
print("bphistar (w=0.3): {:.2f}".format(nvphi_03[max_idx_03,0]))
print("bphistar (w=0.9): {:.2f}".format(nvphi_09[max_idx_09,0]))

avg_n_03 = nvphi_03[max_idx_03,1]
avg_n_09 = nvphi_09[max_idx_09,1]

# PLOT THETA V N
plt.close('all')
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(dat_03[:,0], dat_03[:,1], '-o')
ax.plot(dat_09[:,0], dat_09[:,1], '-o')

ax.plot([0, dat_09[:,0].max()], [expt_theta, expt_theta], 'k--')


ax.axvline(avg_n_09, color='orange')
ax.axvline(avg_n_03, color='blue')

ax.set_xlim(-1,325)
plt.savefig("/Users/nickrego/Desktop/theta_v_n.png", transparent=True)
plt.close('all')

plt.plot([0,0], [1,1], '-o', label=r'$w=0.3 \; \mathrm{nm}$')
plt.plot([0,0], [1,1], '-o', label=r'$w=0.9 \; \mathrm{nm}$')
plt.legend()
plt.xlim(100,200)

plt.savefig("/Users/nickrego/Desktop/legend.png", transparent=True)

plt.close('all')

## PLOT MSE v N
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(mse_03[:,0], np.sqrt(mse_03[:,1])/10., '-o')
ax.plot(mse_09[:,0], np.sqrt(mse_09[:,1])/10., '-o')

ax.axvline(avg_n_09, color='orange')
ax.axvline(avg_n_03, color='blue')

ax.set_xlim(-1,325)
plt.savefig("/Users/nickrego/Desktop/r2_v_n.png", transparent=True)
plt.close('all')


