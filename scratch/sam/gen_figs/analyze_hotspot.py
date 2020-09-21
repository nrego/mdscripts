
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

    #avg_delta_f = (new_energies_up.sum() + new_energies_down.sum()) / (k_c+k_o)
    #abs_delta_f = (np.abs(new_energies_up).sum() + np.abs(new_energies_down).sum()) / (k_c+k_o)

    # np->po
    if new_energies_up.size:
        avg_energy_up = new_energies_up.mean()
    else:
        assert k_c == 0
        avg_energy_up = 0
    assert avg_energy_up >= 0

    # po->np
    if new_energies_down.size:
        avg_energy_down = new_energies_down.mean()
    else:
        assert k_o == 0
        avg_energy_down = 0
    assert avg_energy_down <= 0

    #return pt_idx, avg_delta_f, abs_delta_f

    return pt_idx, avg_energy_up, avg_energy_down


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

bins_tot = np.unique(np.append(bins_down, bins_up))

max_norm = np.round(bins_tot.max()/alpha_not, 2)
bins_norm = np.arange(-max_norm, max_norm+0.001, 0.0005)

# Hist of counts of avg energy for each ko
tot_avg_up_energy_array = np.zeros((len(fnames), bins_tot.size-1))
tot_avg_down_energy_array = np.zeros_like(tot_avg_up_energy_array)

tot_avg_energy_array = np.zeros_like(tot_avg_up_energy_array)
tot_abs_energy_array = np.zeros_like(tot_avg_up_energy_array)

# Same thing, but normalized by pattern f
norm_avg_up_energy_array = np.zeros((len(fnames), bins_norm.size-1))
norm_avg_down_energy_array = np.zeros_like(norm_avg_up_energy_array)

norm_avg_energy_array = np.zeros_like(norm_avg_up_energy_array)
norm_abs_energy_array = np.zeros_like(norm_avg_up_energy_array)

tot_avg_up_states = np.zeros_like(tot_avg_up_energy_array).astype(object)
tot_avg_down_states = np.zeros_like(tot_avg_up_energy_array).astype(object)
tot_avg_states = np.zeros_like(tot_avg_energy_array).astype(object)
tot_abs_states = np.zeros_like(tot_abs_energy_array).astype(object)

norm_avg_up_states = np.zeros_like(norm_avg_up_energy_array).astype(object)
norm_avg_down_states = np.zeros_like(norm_avg_up_energy_array).astype(object)
norm_avg_states = np.zeros_like(norm_avg_energy_array).astype(object)
norm_abs_states = np.zeros_like(norm_abs_energy_array).astype(object)

k_o_vals = np.zeros(len(fnames))


for i, fname in enumerate(fnames[:3]):

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
            pt, avg_energy_up, avg_energy_down = future.get_result(discard=True)
            state = State(pt)
            base_f = alpha_not + alpha_k_o*state.k_o + alpha_n_oo*state.n_oo + alpha_n_oe*state.n_oe

            avg_energy = (avg_energy_up*state.k_c + avg_energy_down*state.k_o) / state.N
            abs_energy = (np.abs(avg_energy_up)*state.k_c + np.abs(avg_energy_down)*state.k_o) / state.N
            
            norm_up = avg_energy_up / base_f
            norm_down = avg_energy_down / base_f
            norm_avg = avg_energy / base_f
            norm_abs = abs_energy / base_f

            bin_idx_avg_up = np.digitize(avg_energy_up, bins_tot) - 1
            bin_idx_avg_down = np.digitize(avg_energy_down, bins_tot) - 1
            bin_idx_avg = np.digitize(avg_energy, bins_tot) - 1
            bin_idx_abs = np.digitize(abs_energy, bins_tot) - 1

            bin_idx_norm_up = np.digitize(norm_up, bins_norm) - 1
            bin_idx_norm_down = np.digitize(norm_down, bins_norm) - 1
            bin_idx_norm_avg = np.digitize(norm_avg, bins_norm) - 1
            bin_idx_norm_abs = np.digitize(norm_abs, bins_norm) - 1

            tot_avg_up_energy_array[k_o, bin_idx_avg_up] += 1
            tot_avg_down_energy_array[k_o, bin_idx_avg_down] += 1
            tot_avg_energy_array[k_o, bin_idx_avg] += 1
            tot_abs_energy_array[k_o, bin_idx_abs] += 1

            norm_avg_up_energy_array[k_o, bin_idx_norm_up] += 1
            norm_avg_down_energy_array[k_o, bin_idx_norm_down] += 1
            norm_avg_energy_array[k_o, bin_idx_norm_avg] += 1
            norm_abs_energy_array[k_o, bin_idx_norm_abs] += 1


            if not tot_avg_up_states[k_o, bin_idx_avg_up]:
                tot_avg_up_states[k_o, bin_idx_avg_up] = list()
            if not tot_avg_down_states[k_o, bin_idx_avg_down]:
                tot_avg_down_states[k_o, bin_idx_avg_down] = list()
            if not tot_avg_states[k_o, bin_idx_avg]:
                tot_avg_states[k_o, bin_idx_avg] = list()
            if not tot_abs_states[k_o, bin_idx_abs]:
                tot_abs_states[k_o, bin_idx_abs] = list()

            if not norm_avg_up_states[k_o, bin_idx_norm_up]:
                norm_avg_up_states[k_o, bin_idx_norm_up] = list()
            if not norm_avg_down_states[k_o, bin_idx_norm_down]:
                norm_avg_down_states[k_o, bin_idx_norm_down] = list()
            if not norm_avg_states[k_o, bin_idx_norm_avg]:
                norm_avg_states[k_o, bin_idx_norm_avg] = list()
            if not norm_abs_states[k_o, bin_idx_norm_abs]:
                norm_abs_states[k_o, bin_idx_norm_abs] = list()

            tot_avg_up_states[k_o, bin_idx_avg_up].append(pt)
            tot_avg_down_states[k_o, bin_idx_avg_down].append(pt)
            tot_avg_states[k_o, bin_idx_avg].append(pt)
            tot_abs_states[k_o, bin_idx_abs].append(pt)

            norm_avg_up_states[k_o, bin_idx_norm_up].append(pt)
            norm_avg_down_states[k_o, bin_idx_norm_down].append(pt)
            norm_avg_states[k_o, bin_idx_norm_avg].append(pt)
            norm_abs_states[k_o, bin_idx_norm_abs].append(pt)



    k_o_vals[i] = k_o
    end_time = time.time()

    print("   time: {:.2f}".format(end_time-start_time))
    sys.stdout.flush()

np.savez_compressed('hotspot_anal.dat', bins_tot=bins_tot, bins_norm=bins_norm,
                    tot_avg_energy_array=tot_avg_energy_array, tot_abs_energy_array=tot_abs_energy_array,
                    tot_avg_up_energy_array=tot_avg_up_energy_array, tot_avg_down_energy_array=tot_avg_down_energy_array,
                    norm_avg_energy_array=norm_avg_energy_array, norm_abs_energy_array=norm_abs_energy_array,
                    norm_avg_up_energy_array=norm_avg_up_energy_array, norm_avg_down_energy_array=norm_avg_down_energy_array,
                    tot_avg_up_states=tot_avg_up_states, tot_avg_down_states=tot_avg_down_states,
                    tot_avg_states=tot_avg_states, tot_abs_states=tot_abs_states
                    )

