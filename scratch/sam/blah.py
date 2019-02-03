from __future__ import division

import numpy as np
import MDAnalysis

from mdtools import MDSystem

import glob, os
from constants import k

from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

COLOR_CONTACT = '#7F00FF' # purple
COLOR_NO_CONTACT = COLOR_NOT_PRED = COLOR_TN = '#7F7F7F' # gray
COLOR_PHIL = '#0560AD' # blue2
COLOR_PHOB = '#D7D7D7' # light gray
COLOR_PRED = '#FF7F00' # orange 
COLOR_TP = '#FF007F' # pink
COLOR_FP = '#7F3F00' # dark orange
COLOR_FN = '#3F007F' # dark purple

def make_piechart(slices, colors, ecolors, outname, showtext=True, groups=[[0,1],[2,3]], radfraction=0.2, transparent=False):
    assert len(groups) == 2
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5,5))
    wedgeprops = {'edgecolor':'k', 'linewidth':6}

    patches, texts, pcts = ax.pie(slices, labeldistance=1.15, colors=colors, autopct=lambda p: '{:1.1f}\%'.format(p), wedgeprops=wedgeprops)
    [pct.set_fontsize(20) for pct in pcts]
    ax.axis('equal')

    for patch, ecolor in zip(patches, ecolors):
        patch.set_edgecolor(ecolor)

    fig.savefig('{}'.format(outname), transparent=True)
    plt.close('all')


    fig, ax = plt.subplots(figsize=(5,5))
    ax.axis('equal')
    ax.axis('off')
    #embed()
    i = groups[0]
    ang = np.deg2rad((patches[i[-1]].theta2 + patches[i[0]].theta1)/2.0)
    wedges = []
    for j in i:
        patch = patches[j]
        center = (radfraction*patch.r*np.cos(ang), radfraction*patch.r*np.sin(ang))
        wedges.append(mpatches.Wedge(center, patch.r, patch.theta1, patch.theta2, edgecolor=patch.get_edgecolor(), facecolor=patch.get_facecolor()))
        text_ang = np.deg2rad((patch.theta1 + patch.theta2)/2.0)
        pct = pcts[j].get_text()
        if slices[j] == 0:
            pct = ''
            fontsize = 10
        elif slices[j] < 0.01:
            fontsize = 10
        elif slices[j] < 0.1:
            fontsize = 15
        else:
            fontsize = 20
        if text_ang > np.pi and text_ang < (3*np.pi)/2.0:
            align='top'
            text_pos = np.array(((1.12)*patch.r*np.cos(text_ang), (1.12)*patch.r*np.sin(text_ang))) + np.array(center)
        else:
            align='baseline'
            text_pos = np.array(((1.08)*patch.r*np.cos(text_ang), (1.08)*patch.r*np.sin(text_ang))) + np.array(center)
        if showtext:
            ax.text(*text_pos, s=pct, fontsize=fontsize, verticalalignment=align)
    i = groups[1]
    for j in i:
        try:
            patch = patches[j]
        except:
            embed()
        center = patch.center
        
        wedges.append(mpatches.Wedge(center, patch.r, patch.theta1, patch.theta2, edgecolor=patch.get_edgecolor(), facecolor=patch.get_facecolor()))
        text_ang = np.deg2rad((patch.theta1 + patch.theta2)/2.0)
        pct = pcts[j].get_text()
        if slices[j] == 0:
            pct = ''
            fontsize = 10
        elif slices[j] < 0.01:
            fontsize = 10
        elif slices[j] < 0.1:
            fontsize = 15
        else:
            fontsize = 20
        if text_ang > np.pi and text_ang < (3*np.pi)/2.0:
            align='top'
            text_pos = np.array(((1.12)*patch.r*np.cos(text_ang), (1.12)*patch.r*np.sin(text_ang))) + np.array(center)
        else:
            align='baseline'
            text_pos = np.array(((1.08)*patch.r*np.cos(text_ang), (1.08)*patch.r*np.sin(text_ang))) + np.array(center)
        if showtext:
            ax.text(*text_pos, s=pct, fontsize=fontsize, verticalalignment=align)

    collection = PatchCollection(wedges)
    collection.set_facecolors(colors)
    collection.set_linewidths(6)
    collection.set_edgecolors(ecolors)

    #collection.set_array(np.array(colors))
    ax.add_collection(collection)
    ax.autoscale(True)
    #ax.text(*pcts[0].get_position(), s=pcts[0].get_text())

    fig.savefig('{}_nolabel'.format(outname), transparent=transparent)

    plt.close('all')

beta = 1/(300*k)
thresh = 0.5

fnames = sorted(glob.glob('../phi_*/rho_data_dump_rad_6.0.dat.npz'))

phi_vals = [ fname.split('/')[1].split('_')[-1] for fname in fnames ]

sys = MDSystem('../phi_000/confout.gro', '../phi_000/confout.gro', sel_spec='resid 4435-4470 and (name S*)')
prot = sys.prot
hydropathy_mask = (prot.resnames == 'CH3')
k = hydropathy_mask.sum()
print('k={} ({:0.2f})'.format(k, k/36))

n_0 = np.load('../phi_000/rho_data_dump_rad_6.0.dat.npz')['rho_water'].mean(axis=0)

n_surf = 36

nvphi = np.loadtxt('../ntwid_out.dat')
max_idx = np.argmax(nvphi[:,2])
betaphimax = nvphi[max_idx, 0]

print("beta phi_star: {}".format(betaphimax))
for fname, phi_val in zip(fnames, phi_vals):

    this_phi = float(phi_val)/10.0 #* beta
    this_beta_phi = int(np.round(beta*float(phi_val))) * 10
    print('phi: {:0.2f}'.format(this_phi))

    n_phi = np.load(fname)['rho_water'].mean(axis=0)

    rho_phi = n_phi/n_0

    dewet_mask = pred_mask = (rho_phi < thresh)
    n_dewet = dewet_mask.sum()
    dewet_phob = (dewet_mask & hydropathy_mask).sum()
    dewet_phil = (dewet_mask & ~hydropathy_mask).sum()

    print('  n_dewet: {}  frac_phob: {:0.2f}'.format(n_dewet, dewet_phob/n_dewet))

    prot.tempfactors = rho_phi
    prot.write('phi_{}_struct.pdb'.format(phi_val))

    pred_phob = (pred_mask & hydropathy_mask).sum()/n_surf
    pred_phil = (pred_mask & ~hydropathy_mask).sum()/n_surf
    no_pred_phob = (~pred_mask & hydropathy_mask).sum()/n_surf
    no_pred_phil = (~pred_mask & ~hydropathy_mask).sum()/n_surf

    colors = (COLOR_PRED, COLOR_NOT_PRED, COLOR_NOT_PRED, COLOR_PRED)
    ecolors = (COLOR_PHOB, COLOR_PHOB, COLOR_PHIL, COLOR_PHIL)
    #make_piechart([pred_phob, no_pred_phob, no_pred_phil, pred_phil], colors, ecolors, '{}_pred'.format(args.outpath), showtext=False)
    make_piechart([pred_phob, no_pred_phob, no_pred_phil, pred_phil], colors, ecolors, '{:03d}_pred'.format(this_beta_phi))