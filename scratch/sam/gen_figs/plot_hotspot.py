
import numpy as np

import argparse
from IPython import embed
import os, glob

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from scipy.special import binom

import itertools

from scratch.sam.util import *

import sys
from scratch.sam.util import GetDelta
from scratch.neural_net.lib import mymap  

import pickle

import shutil

from work_managers.environment import default_env
import time

from work_managers import _available_work_managers
wm = default_env.make_work_manager()
#wm = _available_work_managers['serial'].from_environ()
norm = plt.Normalize(-7,7)
# This dictionary defines the colormap
cdict = {'red':  ((0.0, 0.6, 0.6),   # no red at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.6, 0.6)),  # set to 0.8 so its not too bright at 1

        'green': ((0.0, 0.6, 0.6),   # set to 0.8 so its not too bright at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0)),  # no green at 1

        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0))   # no blue at 1
       }

# colors at the limits
col_dn = (0.6, 0.6, 0.0, 1.0)
col_up = (0.6, 0.0, 0.0, 1.0)

# Create the colormap using the dictionary
GnRd = mpl.colors.LinearSegmentedColormap('GnRd', cdict)
GnRd.set_bad(color=mymap(0.5))
def plot_colorbar(norm=norm, cmap=GnRd):
    plt.close()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ticks=[norm.vmin, 0,  norm.vmax])
    cbar.ax.set_yticklabels([])
    plt.axis('off')

#plot_colorbar()

# Normalize a k_o x n_bins density matrix at each k_o value
# Return norm matrix
def norm_density(density):
    tots = density.sum(axis=1)
    density /= tots[:,None]


# Run from sam_data/


def color_state_by_hotspot(state, delta, norm=norm, cmap=GnRd):
    plt.close()
    new_states_up, new_energies_up = delta(state.methyl_mask, state.pt_idx)
    new_states_down, new_energies_down = delta(state.methyl_mask, state.avail_indices)

    all_energies = np.zeros(state.N)
    all_energies[state.pt_idx] = new_energies_up
    all_energies[state.avail_indices] = new_energies_down

    feat = np.zeros(state.pos_ext.shape[0])
    #feat[:] = np.nan
    feat[state.patch_indices[state.methyl_mask]] = new_energies_up
    feat[state.patch_indices[~state.methyl_mask]] = new_energies_down
    feat = plot_feat(feat, state.p+2, state.q+2)
    #feat[0] = np.nan
    
    avg_delta_f = (new_energies_up.sum() + new_energies_down.sum()) / (state.N)
    abs_delta_f = (np.abs(new_energies_up.sum()) + np.abs(new_energies_down.sum())) / (state.N)
    
    print("average delta f: {:.2f}".format(avg_delta_f))
    print("  abs delta f: {:.2f}".format(abs_delta_f))
    edge_mask = [0,1,2,3,4,5,6,7,8,15,16,23,24,31,32,39,40,47,48,55,56,57,58,59,60,61,62,63]
    #embed()
    plot_hextensor(feat, norm=norm, cmap=cmap, mask=edge_mask)

N = 36


mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'lines.linewidth':2})
mpl.rcParams.update({'lines.markersize':12})
mpl.rcParams.update({'legend.fontsize':10})

states = np.load("data/sam_pattern_06_06.npz")["states"]
reg = np.load("data/sam_reg_m3.npy").item()
reg_meth = np.load("data/sam_reg_m3_meth.npy").item()

alpha_k_o, alpha_n_oo, alpha_n_oe = reg.coef_
alpha_k_c, alpha_n_cc, alpha_n_ce = reg_meth.coef_

state = State(np.array([], dtype=int))
delta = GetDelta(state.adj_mat, state.ext_count, alpha_k_c, alpha_n_cc, alpha_n_ce)


ds = np.load('data/hotspot_anal.dat.npz')

# Up: phobic->philic hotspots
bins_up = ds['bins_up']
# Down: philic->phobic hotspots
bins_down = ds['bins_down']

# For average, abs sus with k_o
bins_tot = ds['bins_tot']

#######################################################

k_o_vals = np.arange(N+1)


density_up_down = ds['density_up_down']

# np->po 
# Shape: (k_o, df_up_vals)
density_up = density_up_down.sum(axis=2)
norm_density(density_up)
#po->np
density_down = density_up_down.sum(axis=1)
norm_density(density_down)

# Pure polar and non-polar patches have no up or down densities, resp.
density_up[-1] = np.nan
density_down[0] = np.nan

ens_avg_up = np.dot(density_up, bins_up)
ens_avg_down = np.dot(density_down, bins_down)
ens_var_up = np.dot(density_up, bins_up**2) - ens_avg_up**2
ens_var_down = np.dot(density_down, bins_down**2) - ens_avg_down**2

density_avg = ds['density_avg']
density_abs = ds['density_abs']

norm_density(density_avg)
norm_density(density_abs)

ens_avg_avg = np.dot(density_avg, bins_tot)
ens_avg_abs = np.dot(density_abs, bins_tot)
ens_var_avg = np.dot(density_avg, bins_tot**2) - ens_avg_avg**2
ens_var_abs = np.dot(density_abs, bins_tot**2) - ens_avg_abs**2

states_up_down = ds['states_up_down']
states_avg = ds['states_avg']
states_abs = ds['states_abs']

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
color_state_by_hotspot(state, delta, norm=norm)
plt.savefig('/Users/nickrego/Desktop/fig_bycolor', transparent=True)

plot_colorbar()

plt.savefig('/Users/nickrego/Desktop/colorbar', transparent=True)
plt.close()

## Just plot external polar border
feat = state.feat
plot_hextensor(feat, mask=state.patch_indices)
plt.savefig('/Users/nickrego/Desktop/edge', transparent=True)
plt.close()

## Plot a state and its hotspot map
def plot_it(state, delta, label='', norm=norm):
    plt.close()
    state.plot()
    plt.savefig("/Users/nickrego/Desktop/fig_{}".format(label), transparent=True)
    color_state_by_hotspot(state, delta, norm=norm)
    plt.savefig("/Users/nickrego/Desktop/fig_color_{}".format(label), transparent=True)

#norm = plt.Normalize(-6, 6)

# Plot sample configs with min, mid, and max df_abs at 
#   selected k_o vals
for idx in [9,18,27]:
    print("\nko: {}".format(idx))
    dos = density_abs[idx]
    pts = states_abs[idx]
    
    # These states are occupied...
    mask = dos > 0
    midpt = mask.sum() // 2
    # Most resistant to mutations
    min_pts = pts[mask][0]
    # Most susceptible to mutations
    max_pts = pts[mask][-1]
    mid_pts = pts[mask][midpt]

    min_state = State(min_pts[0])
    max_state = State(max_pts[0])
    mid_state = State(mid_pts[0])

    plot_it(min_state, delta, label='{:02d}_min'.format(idx), norm=norm)
    #plot_it(mid_state, delta, label='{:02d}_mid'.format(idx), norm=norm)
    plot_it(max_state, delta, label='{:02d}_max'.format(idx), norm=norm)


plt.close('all')

#max_val = 7

fig, ax = plt.subplots(figsize=(5.5,5))
#ax.plot(k_o_vals, ens_avg_up, '-o', color=col_up)
#ax.plot(k_o_vals, -ens_avg_down, '-o', color=col_dn)
ax.plot(k_o_vals, ens_avg_abs, '-o', color='k')

#ax.set_ylim(-max_val, max_val)
ax.set_xticks([0,9,18,27,36])
fig.tight_layout()
plt.savefig("/Users/nickrego/Desktop/df_v_ko", transparent=True)


mod = linear_model.LinearRegression()
mod.fit(k_o_vals.reshape(-1,1), ens_avg_abs)
