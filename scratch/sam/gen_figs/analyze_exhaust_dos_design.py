
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

import pickle

import shutil

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 40})
mpl.rcParams.update({'ytick.labelsize': 40})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'lines.linewidth':2})
mpl.rcParams.update({'lines.markersize':12})
mpl.rcParams.update({'legend.fontsize':10})

COLOR_BREAK = '#1f77b4'
COLOR_BUILD = '#7f7f7f'

np.random.seed()

homedir = os.environ['HOME']
# If true, plot a config (build, break) at each ko
plot_state = False


def plot_it(ko, methyl_mask_break, methyl_mask_build):

    state_break = State(np.arange(36)[methyl_mask_break].astype(int))
    state_build = State(np.arange(36)[methyl_mask_build].astype(int))

    plt.close('all')
    state_break.plot()
    plt.savefig("{}/Desktop/state_break_ko_{:02d}".format(homedir, ko), transparent=True)

    plt.close('all')
    state_build.plot()
    plt.savefig("{}/Desktop/state_build_ko_{:02d}".format(homedir, ko), transparent=True)

    plt.close('all')

## After plotting something (e.g. f or n_oo) vs ko (from greedy),
##   fill in intermediate points and color according to WL-sampled dos.
##
## 'hist' is actually log(dos), so entropies
#.   if 'constr' is True, constrain at each ko
#
def fill_in_wl(n, hist, vals, constr=False):

    # Log of total number of dos for each ko
    log_omega = np.zeros(n+1)

    for i in range(n+1):
        log_omega[i] = np.log(binom(n, i))

    kk, vv = np.meshgrid(np.arange(n+1)-0.5, vals, indexing='ij')

    if not constr:
        plt.pcolormesh(kk, vv, hist, cmap='jet', norm=plt.Normalize(0,20))

    norm_cst = np.log(np.exp(hist).sum())
    max_val = hist.max() - norm_cst

    if constr:
        for i in range(n+1):
            this_hist = hist[i]
            ones = np.ones_like(this_hist)*i
            norm = np.log(np.exp(this_hist).sum())
            occ = this_hist >= 0

            blah = this_hist - norm_cst
        
            plt.scatter(ones[occ], vals[occ], marker='.', cmap='jet', c=blah[occ], norm=plt.Normalize(-20, 0))
    
    plt.colorbar()

    ax = plt.gca()
    ax.set_xticks([0,9,18,27,36])




## Plot greedy design stuff.
## 
## FIRST: run 'extract_dos' from sam_dos/ to extract p=q=6 SAM dos into 'sam_dos.npz'
#
#. SECOND: Run 'extract_dos_energies' from small_patterns to construct log(dos) over energies, noo, and noe,
#.    all as fn's of ko (OUTPUT: 'sam_dos_f_min_max.npz')
#
#. FINALLY: Run this from 'sam_data'
#

pq_vals = [(4,4),(2,18), (3,12), (4, 9), (6,6), (10,10), (5, 20), (4, 25), (2, 50), (20,20)]
ds_design = np.load("data/greedy_design_trials.npz")

all_hyster = np.zeros(len(pq_vals), dtype=object)
all_xstar = np.zeros(len(pq_vals), dtype=float)
all_hstar = np.zeros_like(all_xstar)

for i_pq, (p,q) in enumerate(pq_vals):

    print("p: {} q: {}".format(p,q))
    n = p*q

    ko = np.arange(n+1)

    indices = np.arange(n)

    dummy_state = State(indices, p=p, q=q)
    state_po = State(np.array([], dtype=int), p=p, q=q)
    adj_mat = dummy_state.adj_mat
    ext_count = dummy_state.ext_count

    max_n_internal = dummy_state.n_cc
    max_n_external = dummy_state.n_ce
    tot_edges = max_n_internal + max_n_external

    reg = np.load("data/sam_reg_m3.npy").item()

    # ko, noo, noe
    alpha1, alpha2, alpha3 = reg.coef_

    alpha_kc = -alpha1 - 6*alpha2
    alpha_n_cc = alpha2
    alpha_n_ce = alpha2 - alpha3


    e_min = 0
    e_max = state_po.k_o*alpha1 + state_po.n_oo*alpha2 + state_po.n_oe*alpha3


    de = 0.1
    e_bins = np.arange(0, 400, 0.1)

    header = 'p_{:02d}_q_{:02d}'.format(p, q)
    
    design_f_build = ds_design['all_f_build'].item()[header]
    design_f_break = ds_design['all_f_break'].item()[header]

    ds = np.load('data/sam_dos_f_min_max.npz')
    min_f = ds['min_f']
    max_f = ds['max_f']
    #vals_pq = ds['vals_pq']
    vals_ko = ds['vals_ko']
    vals_f = ds['vals_f']
    vals_noo = ds['vals_noo']
    vals_noe = ds['vals_noe']
    ener_hist = ds['ener_hist']
    noo_hist = ds['noo_hist']
    noe_hist = ds['noe_hist']

    #i_pq = np.where((vals_pq[:,0]==p) & (vals_pq[:,1] == q))[0].item()
    plt.close('all')

    avg_f_break = design_f_break.mean(axis=0)
    avg_f_build = design_f_build.mean(axis=0)

    # f 
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6,6))

    plt.plot(ko, avg_f_break, '-o', color=COLOR_BREAK, markeredgecolor='k')
    plt.plot(ko, avg_f_build, '-o', color=COLOR_BUILD, markeredgecolor='k')
    #ax.set_xticks([0,9,18,27,36])

    #fill_in_wl(n, ener_hist[i_pq, :n+1], vals_f, constr=True)
    ax.set_ylim(-1, e_max+10)
    fig.tight_layout()
    plt.show()
    plt.savefig('{}/Desktop/avg_e_p_{:02d}_q_{:02d}'.format(homedir, p, q), transparent=True)

    plt.close('all')


    ## Hysteresis plots ##

    hyster_f = (avg_f_break - avg_f_build) / e_max
    fig, ax = plt.subplots(figsize=(6,6))

    plt.plot(ko/n, hyster_f, 'k-o')
    plt.tight_layout()
    plt.show()
    plt.savefig('{}/Desktop/hyster_p_{:02d}_q_{:02d}'.format(homedir, p, q))
    plt.close('all')


    this_hyst = np.vstack((ko/n, hyster_f)).T  
    max_idx = np.argmax(this_hyst[:,1])
    all_hyster[i_pq] = this_hyst

    all_xstar[i_pq] = this_hyst[max_idx,0]
    all_hstar[i_pq] = this_hyst[max_idx,1]


    del design_f_build, design_f_break 

plt.close()
#for i_pq in [0,1,2,3]:
for i_pq in [5,6,7,8]:
#for i_pq in [4,5,9]:
    p,q = pq_vals[i_pq]
    print("p: {} q: {}".format(p,q))

    x = all_hyster[i_pq]
    plt.plot(x[:,0], x[:,1], 'o-', label='{} by {}'.format(p,q))

plt.legend()

plt.close()

#plt.scatter()

## Plot out trajectory
p = 2
q = 50
n = p*q
indices = np.arange(n, dtype=int)
header = 'p_{:02d}_q_{:02d}'.format(p, q)

states_break = ds_design['all_states_break'].item()[header]
states_build = ds_design['all_states_build'].item()[header]

i_trial = 100

this_build = states_build[i_trial]
this_break = states_break[i_trial]

del states_break, states_build


for i_ko in range(n+1):
    print("doing i_ko: {}".format(i_ko))
    plt.close('all')

    state_build = State(indices[this_build[i_ko].astype(bool)], p=p, q=q)
    state_break = State(indices[this_break[i_ko].astype(bool)], p=p, q=q)

    state_build.plot()
    plt.savefig('{}/Desktop/build_p_{:02d}_q_{:02d}_rd_{:04d}'.format(homedir, p, q, n-i_ko))
    plt.close('all')
    state_break.plot()
    plt.savefig('{}/Desktop/break_p_{:02d}_q_{:02d}_rd_{:04d}'.format(homedir, p, q, i_ko))
    plt.close('all')

