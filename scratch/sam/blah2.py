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
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


ds = np.load('analysis_data.dat.npz')

energies = ds['energies']
rms_bins = ds['rms_bins']
k_bins = ds['k_bins']

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(rms_bins[:-1], k_bins[:-1])

min_energy = np.nanmin(energies)
max_energy = np.nanmax(energies)
norm = mpl.colors.Normalize(vmin=min_energy, vmax=max_energy)
#surf = ax.plot_surface(X, Y, energies.T, cmap=cm.nipy_spectral, norm=norm)
bottom = np.zeros_like(energies)
dy = 1
dx = 0.05
dz = energies.T.ravel()
mask = np.ma.masked_invalid(dz).mask
vals = np.interp(dz, [min_energy, max_energy], [0, 1])
colors = cm.nipy_spectral(vals)
dz -= min_energy
dz[mask] = 0
erange = max_energy - min_energy
colors[mask] = np.array([1.,1.,1.,0])
ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(dz), dx, dy, dz, shade=True, color=colors)

ax.set_zlim(0, erange)
zticks = ax.get_zticks()
#ax.set_zticks(zticks)
#ax.set_zticklabels(zticks+min_energy)

#ax.set_zlim(min_energy, max_energy)
