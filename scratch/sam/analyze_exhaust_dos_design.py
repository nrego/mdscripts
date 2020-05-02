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

import numpy as np
import sympy

import itertools

from scratch.sam.util import *

import pickle

import shutil

mpl.rcParams.update({'axes.labelsize': 40})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'lines.linewidth':2})
mpl.rcParams.update({'lines.markersize':8})
mpl.rcParams.update({'legend.fontsize':10})

COLOR_BREAK = '#1f77b4'
COLOR_BUILD = '#7f7f7f'

## Analysze entropy from exhaustive search over greedy trajectory design
homedir = os.environ['HOME']

p = 6
q = 6

n = p*q

indices = np.arange(n)

dummy_state = State(indices, ny=p, nz=q)
state_po = State(np.array([], dtype=int))
adj_mat = dummy_state.adj_mat
ext_count = dummy_state.ext_count

reg = np.load("sam_reg_m3.npy").item()

# ko, noo, noe
alpha1, alpha2, alpha3 = reg.coef_

alpha_kc = -alpha1 - 6*alpha2
alpha_n_cc = alpha2
alpha_n_ce = alpha2 - alpha3

e_min = reg.intercept_
e_max = reg.intercept_ + state_po.k_o*alpha1 + state_po.n_oo*alpha2 + state_po.n_oe*alpha3


de = 0.1
e_bins = np.arange(0, 400, 0.1)

headdir = 'p_{:02d}_q_{:02d}'.format(p,q)

n_accessible_states_break = np.zeros(n+1)
n_accessible_states_build = np.zeros(n+1)

avg_e_break = np.zeros(n+1)
avg_e_build = np.zeros(n+1)

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

    x_break = methyl_masks_break.astype(int)
    x_build = methyl_masks_build.astype(int)

    n_cc_break = 0.5*np.linalg.multi_dot((x_break, adj_mat, x_break.T)).diagonal()
    n_cc_build = 0.5*np.linalg.multi_dot((x_build, adj_mat, x_build.T)).diagonal()

    n_ce_break = np.dot(x_break, ext_count)
    n_ce_build = np.dot(x_build, ext_count)

    energy_break = (alpha_kc * kc + alpha_n_cc * n_cc_break + alpha_n_ce * n_ce_break) + e_max
    energy_build = (alpha_kc * kc + alpha_n_cc * n_cc_build + alpha_n_ce * n_ce_build) + e_max



    n_accessible_states_break[i] = n_state_break
    n_accessible_states_build[i] = n_state_build

    ## Get density of states at each kc
    dos_break = np.array([val for val in state_count_break.values()])
    dos_break =  dos_break / dos_break.sum()
    dos_break = dos_break.astype(float)
    dos_build = np.array([val for val in state_count_build.values()])
    dos_build = dos_build / dos_build.sum()
    dos_build = dos_build.astype(float)

    #norm_break = np.log(dos_break.sum())
    #norm_build = np.log(dos_build.sum())

    # log of probability
    s_break = np.log(dos_break) 
    s_build = np.log(dos_build) 

    avg_e_break[i] = np.dot(np.exp(s_break), energy_break)
    avg_e_build[i] = np.dot(np.exp(s_build), energy_build)

    #hist_break, bb = np.histogram(d_energy_break, bins=e_bins, weights=np.exp(s_break))
    #hist_build, bb = np.histogram(d_energy_build, bins=e_bins, weights=np.exp(s_build))

    #e_landscape_break[i] = np.log(hist_break)
    #e_landscape_build[i] = np.log(hist_build)

    #plt.close('all')
    #fig, ax = plt.subplots()
    #ax.bar(bb[:-1], hist_break, color=COLOR_BREAK, width=de)
    #ax.bar(bb[:-1], hist_build, color=COLOR_BUILD, width=de)
    #ax.set_title('ko: {}'.format(n - i))
    #plt.savefig('{}/Desktop/fig_{:02d}.png'.format(homedir, n-i))

assert n_accessible_states_build[0] == n_accessible_states_build[-1] == n_accessible_states_break[0] == n_accessible_states_break[-1] == 1

plt.close('all')
ko = n - np.arange(n+1)
plt.plot(ko, np.log(n_accessible_states_break), '-o', color=COLOR_BREAK)
plt.plot(ko, np.log(n_accessible_states_build), '-o', color=COLOR_BUILD)
plt.savefig('{}/Desktop/n_states_p_{:02d}_q_{:02d}'.format(homedir, p, q), transparent=True)

plt.close('all')

plt.plot(ko, avg_e_break, '-o', color=COLOR_BREAK, markersize=10)
plt.plot(ko, avg_e_build, '-o', color=COLOR_BUILD, markersize=10)
plt.savefig('{}/Desktop/avg_e_p_{:02d}_q_{:02d}'.format(homedir, p, q), transparent=True)

ds = np.load('sam_dos_f_min_max.npz')
min_f = ds['min_f']
max_f = ds['max_f']
f_vals = ds['f_vals']
ener_hist = ds['ener_hist']

#kk, ff = np.meshgrid(ko[::-1]+0.5, f_vals+0.05, indexing='ij')
#plt.pcolormesh(kk, ff, ener_hist, cmap='jet', norm=plt.Normalize(0, ener_hist.max()))
norm_cst = np.log(np.exp(ener_hist).sum())
max_val = ener_hist.max() - norm_cst
for i in range(n+1):
    this_hist = ener_hist[i]
    ones = np.ones_like(this_hist)*i
    norm = np.log(np.exp(this_hist).sum())
    occ = this_hist >= 0

    blah = this_hist - norm_cst
    #print(blah.max())
    plt.scatter(ones[occ], f_vals[occ], marker='.', cmap='jet', c=this_hist[occ], norm=plt.Normalize(0, 20))
plt.colorbar()

ax = plt.gca()
ax.set_xticks([0,9,18,27,36])

