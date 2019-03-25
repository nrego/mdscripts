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
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import networkx as nx
from scipy.spatial import cKDTree
from scipy.special import erf, erfc, erfcinv

from scipy.integrate import cumtrapz

import math
import itertools

#from util import charge_density

i_frame = -2
# Note: alpha = 1/(np.sqrt(2) * sigma) for the sigma of the ewald screening charge
def get_alpha_from_cutoff(rtol, rcut=10.0):

    alpha_r = erfcinv(rtol)
    alpha = alpha_r / rcut

    return alpha


# Electrostatic constant, in units: kJ mol^-1 A e^-2
k_e = 138.9354859 * 10.0
eps = 1/(4*np.pi*k_e)
sqrt_2 = np.sqrt(2)

# Between two point charges
coul = lambda qi, qj, rij: qi*qj/rij

# Point charge and gaussian-screened point charge
# Kappa: inverse width (1/root(2)*width) of screening gaus
def coul_pt(qi, qj, rij, kappa):

    vals = (qi*qj)/(rij) * erfc(kappa*rij)

    return vals

# Between gaussian and point chargeb, or two gaussians
#   Only alpha1 provided: Between point charge and screened gaussian
#   alpha1 and alpha2: Between two gaussians
def coul_gaus(qi, qj, rij, kappa, alpha1, alpha2=None):
    if alpha2 is None:
        alpha = alpha1
    else:
        alpha = (alpha1*alpha2) / np.sqrt(alpha1**2 + alpha2**2)

    vals = (qi*qj)/(rij) * (erf(alpha*rij) - erf(kappa*rij))

    return vals

def get_electro_dir(residues, excl, L, coul_fn, **kwargs):

    e_pot = 0.0

    atms = residues.atoms

    for i in range(atms.n_atoms):
        atm_i = atms[i]

        for j in range(atms.n_atoms):
            atm_j = atms[j]

            if atm_i.id == atm_j.id:
                continue

            # TODO: change this to account for exclusions
            if atm_i.resid == atm_j.resid: #and i in excl and j in excl:
                continue
            
            r_ij = np.linalg.norm(atm_i.position - atm_j.position)
            if r_ij > 10.0:
                continue

            e_pot += coul_fn(atm_i.charge, atm_j.charge, r_ij, **kwargs)

    return e_pot

# Gets the structure factor at this K
def get_electro_recip(atms, m_vec, K):
    S_K = 0.0
    for atm in atms:
        q_i = atm.charge
        r_i = atm.position

        expt = complex(0, np.dot(m_vec*K, r_i))
        S_K += q_i * np.exp(expt)


    return S_K.real**2 + S_K.imag**2


parser = argparse.ArgumentParser('Calculate coulombic potential for frame of simulation')
parser.add_argument('--top', '-s', type=str, default='run.tpr',
                    help='TOP file (default: %(default)s)')
parser.add_argument('--traj', '-f', type=str, default='traj.xtc',
                    help='XTC file (default: %(default)s)')
parser.add_argument('--n-frame', type=int, default=0,
                    help='Frame to calculate potential (default: first frame)')
parser.add_argument('--all-frames', action='store_true',
                    help='If true, calculate potential for all frames')
parser.add_argument('--pt-grp', type=str, default="name _",
                    help='selection string for point charge residues')
parser.add_argument('--gaus-grp', type=str, default="name _",
                    help='selection string for gaussian test charge residues')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Gaussian point charge sigma, in A (default: %(default)s)')
parser.add_argument('--rcut', type=float, default=10.0,
                    help='short-range cutoff, in A (default: %(default)s)')
parser.add_argument('--rtol', type=float, default=1e-5,
                    help='short-range error for determining Ewald screen width (default: %(default)s)')
parser.add_argument('--m-recip', type=int, default=10,
                    help='Number of wave vectors to sum over when calculating the reciprocal Ewald sum')
parser.add_argument('--excl', type=str, default=None,
                    help='List of atomic indices for which to excluded sr interactions (comma sep list)')
args = parser.parse_args()

if args.excl is not None:
    excl_list = [int(idx) for idx in args.excl.split(',')]
else:
    excl_list = []


univ = MDAnalysis.Universe(args.top, args.traj)
atms = univ.atoms
n_atoms = atms.n_atoms

# Group of residues
pt_grp = univ.select_atoms(args.pt_grp).residues
n_pt_atoms = pt_grp.n_atoms

gaus_grp = univ.select_atoms(args.gaus_grp).residues
n_gaus_atoms = gaus_grp.n_atoms

# alpha for screening charge
screen_alpha = get_alpha_from_cutoff(args.rtol, args.rcut)

# alpha for gaussian test charge(s)
test_sig = args.sigma
test_alpha = 1/(np.sqrt(2)*test_sig)

# Assume a cubic box
box = univ.dimensions[:3]
assert box[0] == box[1] == box[2]
L = box[0]
print("box length (nm): {:0.1f}".format(L))

if args.all_frames:
    frames = np.arange(univ.trajectory.n_frames)
else:
    frames = [args.n_frame]

# Total direct, total LR, Pt-Pt SR, G-G SR, Pt-G SR
energy = np.zeros((len(frames), 5))

for i, frame in enumerate(frames):
    print("Frame: {}".format(frame))
    univ.trajectory[frame]

    ## DIRECT SUMS (SR) ##

    # Pt-Pt charge #
    print("Calculating pt-pt coul SR...")
    pot_dir_pt_pt = 0.5*k_e*get_electro_dir(pt_grp, excl_list, L, coul_pt, kappa=screen_alpha)

    energy[i,2] = pot_dir_pt_pt

    print("pt-pt coul SR: {}".format(pot_dir_pt_pt))

    # Gaus-Gaus charge #
    pot_dir_gaus_gaus = 0.5*k_e*get_electro_dir(gaus_grp, [], L, coul_gaus, kappa=screen_alpha, alpha1=test_alpha, alpha2=test_alpha)

    print("gaus-gaus coul SR: {}".format(pot_dir_gaus_gaus))

    energy[i,3] = pot_dir_gaus_gaus

    # pt-Gaus charge #
    pot_dir_pt_gaus = 0

    for atm_i in pt_grp.atoms:
        for atm_j in gaus_grp.atoms:
            rij = np.linalg.norm(atm_i.position - atm_j.position)
            pot_dir_pt_gaus += ((atm_i.charge*atm_j.charge)/rij) * (erf(test_alpha*rij) - erf(screen_alpha*rij))

    pot_dir_pt_gaus *= k_e
    print("pt-gaus coul SR: {}".format(pot_dir_pt_gaus))

    energy[i,4] = pot_dir_pt_gaus

    energy[i,0] = pot_dir_pt_pt + pot_dir_gaus_gaus + pot_dir_pt_gaus

    # Recip vectors
    max_vec = args.m_recip

    m = [np.array(m_vec) for m_vec in itertools.product(range(-max_vec, max_vec+1), repeat=3)]

    # Recip for point charges
    #embed()
    pot_recip = 0
    for m_vec in m:
        if np.array_equal(m_vec, np.zeros(3)):
            continue
        K = (2*np.pi/L)
        k_sq = np.sum((K*m_vec)**2)
        sig = 1/(np.sqrt(2)*screen_alpha)
        pref = np.exp(-0.5*sig*sig*k_sq) / k_sq

        pot_recip += pref*get_electro_recip(atms, m_vec, K)

    V = box.prod()
    pot_recip /= (2*V*eps)

    # Between gaussian and point charge
    pot_self = k_e * (1/(np.sqrt(2*np.pi)*sig)) * np.sum([atm.charge**2 for atm in atms])

    # Between each gaussian and its excl point charges
    pot_excl = 0.0
    for res in pt_grp:
        atms = res.atoms
        for i in excl_list:
            atm_i = atms[i]
            
            for j in excl_list:
                if j == i:
                    continue
                atm_j = atms[j]

                rij = np.linalg.norm(atm_i.position - atm_j.position)
                pot_excl += ((atm_i.charge*atm_j.charge)/rij) * erf(screen_alpha*rij)

    pot_excl = 0.5*k_e*pot_excl
    #embed()

    pot_recip -= pot_self+pot_excl

    print("Coul LR: {}".format(pot_recip))
    energy[i,1] = pot_recip

embed()
header = 'SR   LR    Pt-Pt SR   G-G SR    Pt-G SR'
np.savetxt('energy.dat', energy, header=header)
