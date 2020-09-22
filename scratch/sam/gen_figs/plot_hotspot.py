
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

from work_managers import _available_work_managers
wm = default_env.make_work_manager()
#wm = _available_work_managers['serial'].from_environ()

def plot_colorbar(norm=plt.Normalize(-6,6), cmap=cm.RdGy):
    plt.close()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ticks=[norm.vmin, 0,  norm.vmax])
    cbar.ax.set_yticklabels([])
    plt.axis('off')


def color_state_by_hotspot(state, delta, norm=plt.Normalize(-6, 6), cmap=cm.RdGy):
    plt.close()
    new_states_up, new_energies_up = delta(state.methyl_mask, state.pt_idx)
    new_states_down, new_energies_down = delta(state.methyl_mask, state.avail_indices)

    all_energies = np.zeros(state.N)
    all_energies[state.pt_idx] = new_energies_up
    all_energies[state.avail_indices] = new_energies_down

    feat = np.zeros(state.pos_ext.shape[0])
    feat[state.patch_indices[state.methyl_mask]] = new_energies_up
    feat[state.patch_indices[~state.methyl_mask]] = new_energies_down
    feat = plot_feat(feat, state.p+2, state.q+2)


    avg_delta_f = (new_energies_up.sum() + new_energies_down.sum()) / (state.N)
    abs_delta_f = (np.abs(new_energies_up.sum()) + np.abs(new_energies_down.sum())) / (state.N)
    
    print("average delta f: {:.2f}".format(avg_delta_f))
    print("  abs delta f: {:.2f}".format(abs_delta_f))
    plot_hextensor(feat, norm=norm, cmap=cmap)



states = np.load("sam_data/data/sam_pattern_06_06.npz")['states']

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

state = State(np.array([], dtype=int))
delta = GetDelta(state.adj_mat, state.ext_count, alpha_k_c, alpha_n_cc, alpha_n_ce)

ds = np.load(fnames[0])
bins = ds['bins']
# Up: phobic->philic hotspots
bins_up = bins[0]
# Down: philic->phobic hotspots
bins_down = bins[1]
xx, yy = np.meshgrid(bins_up, bins_down, indexing='ij')


#######################################################
ds = np.load("old_hotspot_anal.dat.npz")

bins_tot = ds['bins_tot']
k_o_vals = 36 - np.arange(37)
k_o_vals = np.append(k_o_vals, -1)

tot_energy = ds['tot_abs_energy_array']
tot_states = ds['tot_abs_states']

xx, yy = np.meshgrid(k_o_vals, bins_tot, indexing='ij')

plt.close('all')


## A: show state and two mutations
state = states[650]
state.plot()
plt.savefig('/Users/nickrego/Desktop/fig.pdf', transparent=True)
new_states_up, new_energies_up = delta(state.methyl_mask, state.pt_idx)
new_states_down, new_energies_down = delta(state.methyl_mask, state.avail_indices)
# phobic hotspot
state_up = State(np.arange(36)[new_states_up[new_energies_up.argmax()]])
state_up.plot()
plt.savefig('/Users/nickrego/Desktop/fig_up.pdf', transparent=True)
state_down = State(np.arange(36)[new_states_down[new_energies_down.argmin()]])
state_down.plot()
plt.savefig('/Users/nickrego/Desktop/fig_down.pdf', transparent=True)

plt.close()
print("min: {:.2f}. max: {:.2f}".format(new_energies_down.min(), new_energies_up.max()))

this_avg_energy = (new_energies_up.sum() + new_energies_down.sum()) / state.N
this_abs_energy = (np.abs(new_energies_up).sum() + np.abs(new_energies_down).sum()) / state.N


## Now plot hotspot map
color_state_by_hotspot(state, delta, norm=plt.Normalize(-7,7))
plt.savefig('/Users/nickrego/Desktop/fig_bycolor', transparent=True)

plot_colorbar()

plt.savefig('/Users/nickrego/Desktop/colorbar', transparent=True)
plt.close()


def plot_it(state, delta, label='', norm=plt.Normalize(-7,7)):
    plt.close()
    state.plot()
    plt.savefig("/Users/nickrego/Desktop/fig_{}".format(label), transparent=True)
    color_state_by_hotspot(state, delta, norm=norm)
    plt.savefig("/Users/nickrego/Desktop/fig_color_{}".format(label), transparent=True)

norm = plt.Normalize(-7, 7)
#for idx in [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]:
for idx in [9,18,27]:
    print("\nko: {}".format(36-idx))
    energies = tot_energy[idx]
    pts = tot_states[idx]
    
    mask = energies > 0
    midpt = mask.sum() // 2
    # Mutation, on average, will make it more phobic
    min_pts = pts[mask][0]
    # Mutation, on average, will make it more philic
    max_pts = pts[mask][-1]
    mid_pts = pts[mask][midpt]

    min_state = State(min_pts[0])
    max_state = State(max_pts[0])
    mid_state = State(mid_pts[0])

    plot_it(min_state, delta, label='{:02d}_min'.format(idx), norm=norm)
    plot_it(mid_state, delta, label='{:02d}_mid'.format(idx), norm=norm)
    plot_it(max_state, delta, label='{:02d}_max'.format(idx), norm=norm)

## Now get average delta f as function of ko
bc = bins_tot[:-1] + 0.5*np.diff(bins_tot)

p_energy = tot_energy / tot_energy.sum(axis=1)[:,None]

## Average energy with each ko
avg_energy = np.dot(p_energy, bc)

plt.close('all')

max_val = 7
k_o_vals = 36 - np.arange(37)
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(k_o_vals, avg_energy, '-o')

ax.set_ylim(-max_val, max_val)
ax.set_xticks([0,9,18,27,36])
fig.tight_layout()
plt.savefig("/Users/nickrego/Desktop/df_v_ko", transparent=True)



