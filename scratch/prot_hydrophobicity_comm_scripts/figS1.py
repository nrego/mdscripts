from __future__ import division, print_function
  

import MDAnalysis
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

import glob, os

import scipy


## Fig S1: How surface hydrophobicity differs if you consider surface residues or atoms

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 30})
mpl.rcParams.update({'ytick.labelsize': 30})
mpl.rcParams.update({'axes.titlesize':40})
mpl.rcParams.update({'legend.fontsize':40})

fnames = glob.glob('*/surf_dat.dat')
labels = []
vals = np.zeros((len(fnames), 8), dtype=float)

for i, fname in enumerate(fnames):
    dirname = os.path.dirname(fname)
    labels.append(dirname)
    vals[i,...] = np.loadtxt(fname)
labels = np.array(labels)

name_lup = {'1brs': 'barnase',
            '1ubq': 'ubiquitin',
            '1qgt': 'capsid',
            '1ubq': 'ubiquitin',
            '1ycr': 'MDM2',
            '253l': 'lysozyme',
            '2b97': 'hydrophobin',
            '3hhp': 'malate\ndehydrogenase'}

order = ['hydrophobin', 'capsid', 'MDM2', 'malate\ndehydrogenase', 'ubiquitin', 'barnase', 'lysozyme']

colors = cm.rainbow(np.linspace(0,1,len(order)))

order_idx = np.zeros(len(order), dtype=int)
for idx, label in enumerate(labels):
    if name_lup[label] in order:
        order_idx[order.index(name_lup[label])] = idx

vals = vals[order_idx,...]
labels = labels[order_idx]

names = [name_lup[label] for label in labels]
dipole = {k:v for k,v in zip(np.loadtxt('dipole.dat', usecols=0, dtype=str), np.loadtxt('dipole.dat', usecols=1))}

n_bars = len(labels)
indices = np.arange(n_bars)
width = 1

gap = 4

plt.cla()
plt.clf()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,8))

# upper left
for idx, name in enumerate(names):
    label = labels[idx]
    n_tot, n_surf, n_res_surf, n_phil_surf, n_phob_surf, n_phob_res, pos_charge_res, neg_charge_res  = vals[idx] 
    this_dipole = dipole[label]

    ax1.bar(indices[idx], n_phob_surf/n_phob_res, width=width, label=name, color=colors[idx])
    ax2.bar(indices[idx], n_surf/n_res_surf, width=width, color=colors[idx])

    ax1.set_xticks([])
    ax1.set_ylim(0,35)
    
    #ax2.set_yticks([])

    ax2.set_xticks([])
    ax2.set_ylim(0,35)
    
    ax3.plot(n_phob_res, n_phob_surf, 'o', color=colors[idx], markersize=18)
    ax4.plot(n_res_surf, n_surf, 'o', color=colors[idx], markersize=18)
    #ax4.bar(indices[idx], n_surf, width=width, label=label, color=colors[idx])
    #ax3.set_xticks([])
    #ax4.set_xticks([])
xvals = np.arange(0,500,1)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(vals[:,5], vals[:,4])

#ax3.plot(vals[:,5], vals[:,4], 'o')
xmin, xmax = ax3.get_xlim()
ymin, ymax = ax3.get_ylim()
ax3.plot(xvals, intercept + slope*xvals, 'k-', zorder=1, linewidth=2)
ax3.set_xlim(xmin,60)
ax3.set_ylim(ymin,1100)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(vals[:,2], vals[:,1])
#ax4.plot(vals[:,2], vals[:,1], 'o')
xmin, xmax = ax4.get_xlim()
ymin, ymax = ax4.get_ylim()
ax4.plot(xvals, intercept + slope*xvals, 'k-', zorder=1, linewidth=2)
ax4.set_xlim(xmin,150)
ax4.set_ylim(ymin,1900)



fig.tight_layout()
fig.subplots_adjust(hspace=0.5, wspace=1.0, left=0.2)
fig.savefig('/Users/nickrego/Desktop/fig.pdf', transparent=True)
plt.show()
plt.clf()
fig, ax = plt.subplots(figsize=(10,10))
for idx, name in enumerate(names):
    label = labels[idx]
    n_tot, n_surf, n_res_surf, n_phil_surf, n_phob_surf, n_phob_res, pos_charge_res, neg_charge_res  = vals[idx] 
    this_dipole = dipole[label]

    ax.bar(indices[idx], n_phob_res/n_res_surf, width=width, label=name, color=colors[idx])

    ax.set_ylim(10,12)
    ax.set_xticks([])
    ax.set_yticks([])



plt.legend(handlelength=1, labelspacing=0.1, framealpha=0)
plt.tight_layout()
plt.savefig('/Users/nickrego/Desktop/label.pdf', transparent=True)
