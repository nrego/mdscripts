import numpy as np
import os
import matplotlib as mpl
import MDAnalysis

def find_contiguous_colors(colors):
    # finds the continuous segments of colors and returns those segments
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg) # the final one
    return segs
 
def plot_multicolored_lines(x,y,colors):
    segments = find_contiguous_colors(colors)
    plt.figure()
    start= 0
    for seg in segments:
        end = start + len(seg)
        l, = ax.plot(x[start:end],y[start:end],lw=2,c=seg[0]) 
        start = end

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 15})
mpl.rcParams.update({'ytick.labelsize': 15})
mpl.rcParams.update({'axes.titlesize':20})
mpl.rcParams.update({'legend.fontsize':10})

dat = np.load("../phi_sims/ni_rad_weighted.dat.npz")
beta_phis = dat['beta_phi']
n_with_phi = dat['avg']
cov_with_phi = dat['cov']
n_0 = n_with_phi[:,0]

buried_mask = n_0 < 5

contact_mask = np.loadtxt("actual_contact_mask.dat", dtype=bool)

rho_with_phi = n_with_phi / n_0[:, None]


## Find when each voxel's beta phi_i^*
beta_phi_star = np.zeros(rho_with_phi.shape[0])
n_phi_vals = beta_phis.size

for i_vox in range(rho_with_phi.shape[0]):
    this_rho = rho_with_phi[i_vox]

    for i_phi in range(n_phi_vals-1,0,-1):
        if this_rho[i_phi] < 0.5:
            continue
        else:
            try:
                this_phi_star = beta_phis[i_phi+1]
            except IndexError:
                this_phi_star = 4
            beta_phi_star[i_vox] = this_phi_star
            break

np.savetxt("beta_phi_star_prot.dat", beta_phi_star)
import pickle
with open('charge_assign.pkl', 'rb') as fin:
    charge_assign = pickle.load(fin)

surf_mask = ~buried_mask

univ = MDAnalysis.Universe("../order.pdb")
surf_atoms = univ.atoms[surf_mask]
surf_rho_with_phi = rho_with_phi[surf_mask]
surf_cov_with_phi = cov_with_phi[surf_mask]
surf_beta_phi_star = beta_phi_star[surf_mask]
surf_contact = contact_mask[surf_mask]

sort_idx = np.argsort(surf_beta_phi_star)

pink = '#FF007F'
orange = '#FF7F00'
gray = '#3F3F3F'
purple = '#7F00FF'

for i, idx in enumerate(sort_idx): 
    print("doing {} out of {}".format(i, len(sort_idx)))
    is_contact = surf_contact[idx]
   

    wet_mask = surf_rho_with_phi[idx,:] >= 0.5

    colors = np.empty_like(wet_mask, dtype=object)
    if is_contact:
        colors[wet_mask] = purple
        colors[~wet_mask] = pink
    else:
        colors[wet_mask] = gray
        colors[~wet_mask] = orange

    plt.close("all") 
    fig, ax = plt.subplots(figsize=(5.5, 5)) 
    bphi = surf_beta_phi_star[idx] 
    atm = surf_atoms[idx] 
    try:
        hyd = charge_assign[atm.resname][atm.name] 
    except KeyError:
        hyd = input("Enter hydrophobicity (1, phob; -1, phil) for atom {} of res {}:\n".format(atm.name, atm.resname))
    
    atm_color = '#D7D7D7' if hyd == 1 else '#0560AD' 
    ax.plot(beta_phis, surf_cov_with_phi[idx]/surf_rho_with_phi[idx,0]) 
    #ax.hlines(0.5, 0, 4) 
    ax.text(0.5, 0.1, 
            r'ATM: {}  RES: {} {}; $\beta \phi_i^*$: {:0.2f}'.format(atm.name, atm.resname, atm.resid, bphi), 
            color=atm_color) 
    ax.set_xlim(0,4) 
    ax.set_ylim(0,20) 
    ax.set_xlabel(r'$\beta \phi$') 
    ax.set_ylabel(r'$\frac{\chi_i}{\langle n_i \rangle_0}$') 
    fig.tight_layout() 
    fig.savefig('sus_{:03d}.pdf'.format(i))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 6)) 

    ax.plot(beta_phis, surf_rho_with_phi[idx]) 
    plot_multicolored_lines(beta_phis, surf_rho_with_phi[idx,:], colors)
    ax.hlines(0.5, 0, 4) 
    ax.text(0.5, 0.1, 
            r'ATM: {}  RES: {} {};  $\beta \phi_i^*$: {:0.2f}'.format(atm.name, atm.resname, atm.resid, bphi), 
            color=atm_color, fontsize=20) 
    ax.set_xlim(0,4) 
    ax.set_ylim(0,1) 
    ax.set_xlabel(r'$\beta \phi$') 
    ax.set_ylabel(r'$\langle \rho_i \rangle_\phi$') 
    fig.tight_layout() 
    fig.savefig('fig_{:03d}.pdf'.format(i))



