
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
import sympy

from scipy.special import binom

import itertools

from scratch.sam.util import *

import sys
from scratch.sam.run_wl_hotspot import GetDelta, reg, reg_meth

import pickle

import shutil

from work_managers.environment import default_env
import time


# Fill 1-D histogram from 2d density field with associated bin assignments for 
#  each position
def fill_hist(hist, this_states, density, sampled_pts, assign):
    for idx, dens, pts in zip(assign.ravel(), density.ravel(), sampled_pts.ravel()):
        hist[idx] += dens

        if pts is not None:
            pts_arr = this_states[idx]
            if pts_arr is None:
                pts_arr = pts.copy()
            else:
                pts_arr = np.vstack((pts_arr, pts))

            this_states[idx] = pts_arr

N = 36
## RUN FROM sam_hotspot_dos/p_6_q_6
##   link in sam_data/
#
## Extracts WL d.o.s. Omega(df_up, df_down) into one place.

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'lines.linewidth':2})
mpl.rcParams.update({'lines.markersize':12})
mpl.rcParams.update({'legend.fontsize':10})

## Run from sam_hotspot_dos/p_6_q_6/

fnames = sorted(glob.glob('hotspot_dos*'))

alpha_k_o, alpha_n_oo, alpha_n_oe = reg.coef_
alpha_k_c, alpha_n_cc, alpha_n_ce = reg_meth.coef_

# Fe of pure phobic patch
alpha_not = reg.intercept_

state = State(np.array([], dtype=int))
delta = GetDelta(state.adj_mat, state.ext_count, alpha_k_c, alpha_n_cc, alpha_n_ce)

ds = np.load(fnames[0])
bins = ds['bins']
# Up: phobic->philic hotspots
bins_up = bins[0]
# Down: philic->phobic hotspots
bins_down = bins[1]

de = np.diff(bins_up)[0]
assert np.allclose(de, -np.diff(bins_down)[-1])

## up and down energies at each position
d_f_up, d_f_down = np.meshgrid(bins_up, bins_down, indexing='ij')

assert bins_up.min() == 0
assert bins_down.max() == 0

bins_tot = np.unique(np.append(bins_down, bins_up))

assert len(fnames) == N + 1

k_o_vals = np.arange(N+1)

# Hist of counts of density of states at each

# Omega_ko(df_up, df_down)
#   Shape: (N+1, bins_up.size-1, bins_down.size-1)
density_up_down = np.zeros((len(fnames), bins_up.size, bins_down.size))
# Sample states at each ko, df_up, df_down
states_up_down = np.zeros_like(density_up_down).astype(object)

density_avg = np.zeros((len(fnames), bins_tot.size))
density_abs = np.zeros_like(density_avg)

states_avg = np.zeros_like(density_avg).astype(object)
states_abs = np.zeros_like(density_abs).astype(object)
states_avg[...] = None
states_abs[...] = None

for i, fname in enumerate(fnames):

    ds = np.load(fname)

    # ...yeah, this is backwards
    #k_c = ds['ko'].item()
    k_c = ds['kc'].item()
    k_o = N - k_c

    df_avg = (k_c*d_f_up + k_o*d_f_down) / N
    df_abs = (k_c*np.abs(d_f_up) + k_o*np.abs(d_f_down)) / N
    assert df_abs.max() <= bins_up.max()

    sampled_pts = ds['sampled_points']
    density = ds['density']
    entropy = ds['entropies']

    mask = density > 0

    assert np.array_equal(bins_up, ds['bins'][0])
    assert np.array_equal(bins_down, ds['bins'][1])

    # Hack-ish fix for k_c=36 or 0; no density
    if k_c == 36:
        ## Integrate out bins_down
        e_up_idx = np.argmax(density.sum(axis=1))
        new_density = np.zeros_like(density)
        new_density[e_up_idx, 0] = 1
        density = new_density

    assert density.sum(axis=0)[-1] == 0
    assert density.sum(axis=1)[-1] == 0

    idx_avg = np.digitize(df_avg, bins_tot) - 1
    idx_abs = np.digitize(df_abs, bins_tot) - 1
    this_density_avg = np.zeros_like(bins_tot)
    this_density_abs = np.zeros_like(bins_tot)
    this_states_avg = np.zeros_like(bins_tot).astype(object)
    this_states_abs = np.zeros_like(bins_tot).astype(object)

    this_states_avg[...] = None
    this_states_abs[...] = None

    fill_hist(this_density_avg, this_states_avg, density, sampled_pts, idx_avg)
    fill_hist(this_density_abs, this_states_abs, density, sampled_pts, idx_abs)


    # All sampled states at this ko,
    #    in (n_states, k_c) shaped array
    #.   all_pts[i] gives pt_idx for ith state
    all_pts = np.concatenate(sampled_pts[mask]).astype(int)

    print("doing ko: {}".format(k_o))
    print("  n unique pts: {}".format(all_pts.shape[0]))

    density_up_down[k_o] = density
    states_up_down[k_o] = sampled_pts

    density_avg[k_o] = this_density_avg
    density_abs[k_o] = this_density_abs

    states_avg[k_o] = this_states_avg
    states_abs[k_o] = this_states_abs




np.savez_compressed('hotspot_anal.dat', bins_up=bins_up, bins_down=bins_down, bins_tot=bins_tot, 
                    density_up_down=density_up_down, states_up_down=states_up_down,
                    density_avg=density_avg, states_avg=states_avg, density_abs=density_abs, states_abs=states_abs)




