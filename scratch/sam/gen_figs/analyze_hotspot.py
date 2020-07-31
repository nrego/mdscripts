
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

# Get average energy change of making a random mutation
#.   For a given state (given by pt_idx, a list of indices of methyls)
def get_avg_mut_energy(pt_idx, delta):
    
    state = State(pt_idx)
    np_indices = pt_idx
    po_indices = state.avail_indices

    k_o = po_indices.size
    k_c = np_indices.size

    assert k_o + k_c == 36

    # Change np group to po
    new_states_up, new_energies_up = delta(state.methyl_mask, np_indices)
    # Change po group to np
    new_states_down, new_energies_down = delta(state.methyl_mask, po_indices)


    return pt_idx, (new_energies_up.sum() + new_energies_down.sum()) / (k_c+k_o)

def task_gen(all_pts, delta):
    
    for pt in all_pts:

        args = (pt, delta)
        kwargs = dict()

        yield get_avg_mut_energy, args, kwargs


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

bins_tot = np.unique(np.append(bins_down, bins_up))

# Hist of counts of avg energy for each ko
tot_energy_array = np.zeros((len(fnames), bins_tot.size-1))
tot_states = np.zeros_like(tot_energy_array).astype(object)


k_o_vals = np.zeros(len(fnames))


for i, fname in enumerate(fnames):

    ds = np.load(fname)

    # ...yeah, this is backwards
    k_c = ds['ko'].item()
    k_o = 36 - k_c
    sampled_pts = ds['sampled_points']
    density = ds['density']
    entropy = ds['entropies']

    mask = density > 0

    assert np.array_equal(bins_up, ds['bins'][0])
    assert np.array_equal(bins_down, ds['bins'][1])

    # All sampled states at this ko,
    #    in (n_states, k_c) shaped array
    #.   all_pts[i] gives pt_idx for ith state
    all_pts = np.concatenate(sampled_pts[mask]).astype(int)

    print("doing ko: {}".format(k_o))
    print("  n unique pts: {}".format(all_pts.shape[0]))

    start_time = time.time()
    with wm:
        for future in wm.submit_as_completed(task_gen(all_pts, delta), queue_size=None):
            pt, avg_delta_f = future.get_result(discard=True)
            bin_idx = np.digitize(avg_delta_f, bins_tot) - 1

            tot_energy_array[i, bin_idx] += 1

            if not tot_states[i, bin_idx]:
                tot_states[i, bin_idx] = list()

            tot_states[i, bin_idx].append(pt)


    end_time = time.time()

    print("   time: {:.2f}".format(end_time-start_time))
    sys.stdout.flush()

np.savez_compressed('hotspot_anal.dat', bins_tot=bins_tot, tot_energy_array=tot_energy_array, 
                    tot_states=tot_states, ko_vals=k_o_vals)

