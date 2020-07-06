
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
#. FINALLY: Run this from 'sam_exhaust_greedy'
#


p = 2
q = 18



n = p*q

indices = np.arange(n)

dummy_state = State(indices, p=p, q=q)
state_po = State(np.array([], dtype=int), p=p, q=q)
adj_mat = dummy_state.adj_mat
ext_count = dummy_state.ext_count

max_n_internal = dummy_state.n_cc
max_n_external = dummy_state.n_ce
tot_edges = max_n_internal + max_n_external

reg = np.load("sam_reg_m3.npy").item()

# ko, noo, noe
alpha1, alpha2, alpha3 = reg.coef_

alpha_kc = -alpha1 - 6*alpha2
alpha_n_cc = alpha2
alpha_n_ce = alpha2 - alpha3


e_min = 0
e_max = state_po.k_o*alpha1 + state_po.n_oo*alpha2 + state_po.n_oe*alpha3


de = 0.1
e_bins = np.arange(0, 400, 0.1)

headdir = 'p_{:02d}_q_{:02d}'.format(p,q)

n_accessible_states_break = np.zeros(n+1)
n_accessible_states_build = np.zeros(n+1)

avg_e_break = np.zeros(n+1)
avg_e_build = np.zeros(n+1)

avg_noo_break = np.zeros_like(avg_e_break)
avg_noo_build = np.zeros_like(avg_e_break)

avg_noe_break = np.zeros_like(avg_e_break)
avg_noe_build = np.zeros_like(avg_e_break)

shutil.copy('{}/break_phob/kc_0000.pkl'.format(headdir), '{}/build_phob/'.format(headdir))
shutil.copy('{}/build_phob/kc_{:04d}.pkl'.format(headdir, n), '{}/break_phob/'.format(headdir))

e_landscape_break = np.zeros((n+1, e_bins.size-1))
e_landscape_build = np.zeros((n+1, e_bins.size-1))
e_landscape_tot = np.zeros_like(e_landscape_build)

XX, YY = np.meshgrid(n-np.arange(n+1), e_bins[:-1], indexing='ij')

for i in range(n+1):
    kc = i
    with open('{}/break_phob/kc_{:04d}.pkl'.format(headdir, i), 'rb') as fin:
        state_count_break = pickle.load(fin)
    with open('{}/build_phob/kc_{:04d}.pkl'.format(headdir, i), 'rb') as fin:
        state_count_build = pickle.load(fin)

    methyl_masks_break = np.array([np.fromstring(state_byte, bool) for state_byte in state_count_break.keys()])
    methyl_masks_build = np.array([np.fromstring(state_byte, bool) for state_byte in state_count_build.keys()])

    
    # Number of unique states at this step
    n_state_break = methyl_masks_break.shape[0]
    n_state_build = methyl_masks_build.shape[0]
    print("i {} n_state break: {}".format(i, n_state_break))

    if plot_state:
        rand_break = np.random.choice(n_state_break)
        rand_build = np.random.choice(n_state_build)
        plot_it(n-kc, methyl_masks_break[rand_break], methyl_masks_build[rand_build])

    x_break = methyl_masks_break.astype(int)
    x_build = methyl_masks_build.astype(int)

    n_cc_break = 0.5*np.linalg.multi_dot((x_break, adj_mat, x_break.T)).diagonal()
    n_cc_build = 0.5*np.linalg.multi_dot((x_build, adj_mat, x_build.T)).diagonal()

    n_ce_break = np.dot(x_break, ext_count)
    n_ce_build = np.dot(x_build, ext_count)


    n_oo_break = max_n_internal - 6*kc + n_cc_break + n_ce_break
    n_oo_build = max_n_internal - 6*kc + n_cc_build + n_ce_build
    n_oe_break = max_n_external - n_ce_break
    n_oe_build = max_n_external - n_ce_build

    energy_break = (alpha_kc * kc + alpha_n_cc * n_cc_break + alpha_n_ce * n_ce_break) + e_max
    energy_build = (alpha_kc * kc + alpha_n_cc * n_cc_build + alpha_n_ce * n_ce_build) + e_max

    tmp = (alpha1 * (n-kc) + alpha2 * n_oo_break + alpha3 * n_oe_break) + e_min
    assert np.allclose(tmp, energy_break)

    n_accessible_states_break[i] = n_state_break
    n_accessible_states_build[i] = n_state_build

    ## Get density of states at each kc
    dos_break = np.array([val for val in state_count_break.values()])
    dos_break =  dos_break / dos_break.sum()
    dos_break = dos_break.astype(float)
    dos_build = np.array([val for val in state_count_build.values()])
    dos_build = dos_build / dos_build.sum()
    dos_build = dos_build.astype(float)



    # log of probability
    s_break = np.log(dos_break) 
    s_build = np.log(dos_build) 

    avg_e_break[i] = np.dot(np.exp(s_break), energy_break)
    avg_e_build[i] = np.dot(np.exp(s_build), energy_build)

    avg_noo_break[i] = np.dot(np.exp(s_break), n_oo_break)
    avg_noo_build[i] = np.dot(np.exp(s_build), n_oo_build)

    avg_noe_break[i] = np.dot(np.exp(s_break), n_oe_break)
    avg_noe_build[i] = np.dot(np.exp(s_build), n_oe_build)

assert n_accessible_states_build[0] == n_accessible_states_build[-1] == n_accessible_states_break[0] == n_accessible_states_break[-1] == 1

ko = n - np.arange(n+1)


ds = np.load('sam_dos_f_min_max.npz')
min_f = ds['min_f']
max_f = ds['max_f']
vals_pq = ds['vals_pq']
vals_ko = ds['vals_ko']
vals_f = ds['vals_f']
vals_noo = ds['vals_noo']
vals_noe = ds['vals_noe']
ener_hist = ds['ener_hist']
noo_hist = ds['noo_hist']
noe_hist = ds['noe_hist']

i_pq = np.where((vals_pq[:,0]==p) & (vals_pq[:,1] == q))[0].item()
plt.close('all')

plt.plot(ko, np.log(n_accessible_states_break), '-o', color=COLOR_BREAK)
plt.plot(ko, np.log(n_accessible_states_build), '-o', color=COLOR_BUILD)
plt.savefig('{}/Desktop/n_states_p_{:02d}_q_{:02d}'.format(homedir, p, q), transparent=True)

## n_oo
plt.close('all')
fig, ax = plt.subplots(figsize=(6,6))

ax.plot(ko, avg_noo_break, '-o', color=COLOR_BREAK, markeredgecolor='k')
ax.plot(ko, avg_noo_build, '-o', color=COLOR_BUILD, markeredgecolor='k')
ax.set_xticks([0,9,18,27,36])
fig.tight_layout()
fill_in_wl(n, noo_hist[i_pq, :n+1], vals_noo, constr=True)
ax.set_ylim(-1, state_po.n_oo+2)
plt.show()
plt.savefig('{}/Desktop/avg_noo_p_{:02d}_q_{:02d}'.format(homedir, p, q), transparent=True)



# n_oe
plt.close('all')
fig, ax = plt.subplots(figsize=(6,6))

plt.plot(ko, avg_noe_break, '-o', color=COLOR_BREAK, markeredgecolor='k')
plt.plot(ko, avg_noe_build, '-o', color=COLOR_BUILD, markeredgecolor='k')
ax.set_xticks([0,9,18,27,36])
fig.tight_layout()
fill_in_wl(n, noe_hist[i_pq, :n+1], vals_noe, constr=True)
ax.set_ylim(-1, state_po.n_oe+2)
plt.show()
plt.savefig('{}/Desktop/avg_noe_p_{:02d}_q_{:02d}'.format(homedir, p, q), transparent=True)



# f 
plt.close('all')
fig, ax = plt.subplots(figsize=(6,6))

plt.plot(ko, avg_e_break, '-o', color=COLOR_BREAK, markeredgecolor='k')
plt.plot(ko, avg_e_build, '-o', color=COLOR_BUILD, markeredgecolor='k')
ax.set_xticks([0,9,18,27,36])

fill_in_wl(n, ener_hist[i_pq, :n+q], vals_f, constr=True)
ax.set_ylim(-1, e_max+10)
fig.tight_layout()
plt.show()
plt.savefig('{}/Desktop/avg_e_p_{:02d}_q_{:02d}'.format(homedir, p, q), transparent=True)



