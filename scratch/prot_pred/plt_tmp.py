import numpy as np
import os

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

homedir = os.environ['HOME']
dat_wham = np.load('pred_reweight/rho_i_with_phi.dat.npz')
dat_nowham = np.load('pred/rho_i_with_phi.dat.npz')

dat_ni_wham = np.load('pred_reweight/ni_dat.dat.npz')
beta_phi_wham = dat_wham['beta_phi']
beta_phi_nowham = dat_nowham['beta_phi']
assert(np.array_equal(beta_phi_wham, dat_ni_wham['beta_phi']))

n_i_wham = dat_ni_wham['avg']
rho_i_wham = dat_wham['avg']
rho_i_nowham = dat_nowham['avg']

buried_mask = np.loadtxt('bound/buried_mask.dat', dtype=bool)
surf_mask = ~buried_mask
actual_contact = np.loadtxt('bound/actual_contact_mask.dat', dtype=bool)[surf_mask]




pink = '#FF007F'
orange = '#FF7F00'
gray = '#3F3F3F'
purple = '#7F00FF'

for idx in range(0, surf_mask.sum(), 10):
    if idx % 100 == 0:
        print("i: {}".format(idx))
    is_contact = actual_contact[idx]


    fig, ax = plt.subplots(figsize=(5.5,5))

    wet_mask = rho_i_wham[idx,:] >= 0.5

    colors = np.empty_like(wet_mask, dtype='|S7')
    if is_contact:
        colors[wet_mask] = purple
        colors[~wet_mask] = pink
    else:
        colors[wet_mask] = gray
        colors[~wet_mask] = orange

    ax.plot(beta_phi_wham, rho_i_wham[idx, :], 'k-')
    plot_multicolored_lines(beta_phi_wham, rho_i_wham[idx,:], colors)
    ax.plot(beta_phi_nowham, rho_i_nowham[idx,:], 'ko')
    ax.hlines(0.5, 0, 4)

    ax.set_xlim(0,4)
    ax.set_ylim(0,1)

    ax.set_xlabel(r'$\beta \phi$')
    ax.set_ylabel(r'$\langle \rho_i \rangle_\phi$')

    fig.tight_layout()
    fig.savefig('{}/Dropbox/prot_prediction_paper/data_reweight/idx_{:04g}.pdf'.format(homedir, idx))

    fig, ax = plt.subplots(figsize=(5.1,5))
    ax.plot(beta_phi_wham, n_i_wham[idx, :], 'k-')
    plot_multicolored_lines(beta_phi_wham, n_i_wham[idx,:], colors)


    ax.set_xlim(0,4)
    ax.set_ylim(0,n_i_wham[idx,0])
    ax.set_xlabel(r'$\beta \phi$')
    ax.set_ylabel(r'$\langle n_i \rangle_\phi$')

    fig.tight_layout()

    fig.savefig('{}/Dropbox/prot_prediction_paper/data_reweight/n_i_idx_{:04g}.pdf'.format(homedir, idx))