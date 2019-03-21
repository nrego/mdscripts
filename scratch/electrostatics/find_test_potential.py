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

i_frame = 0
# Note: alpha = 1/(np.sqrt(2) * sigma) for the sigma of the ewald screening charge
def get_alpha_from_cutoff(rtol, rcoulcut=10.0):

    alpha_r = erfcinv(rtol)
    alpha = alpha_r / rcoulcut

    return alpha


# Electrostatic constant, in units: kJ mol^-1 A e^-2
k_e = 138.9354859 * 10.0
eps = 1/(4*np.pi*k_e)
sqrt_2 = np.sqrt(2)

coul = lambda qi, qj, rij: qi*qj/rij

# Between gaussian screened point charges
def coul_sr(qi, qj, rij, alpha):
    return (qi*qj)/(rij) * erfc(alpha*rij)

# Between gaussian and point charge, or two gaussians
def coul_erf(qi, qj, rij, alpha1, alpha2=None):
    if alpha2 is None:
        alpha = alpha1
    else:
        alpha = (alpha1*alpha2) / np.sqrt(alpha1**2 + alpha2**2)

    lim_zero = 2*alpha/np.sqrt(np.pi)
    vals = (qi*qj)/(rij) * erf(alpha*rij)

    return vals

def get_electro_dir(atms, n_vec, L, coul_fn, **kwargs):

    e_pot = 0.0

    zero_vec = np.array_equal(n_vec, np.zeros(3))

    for i in range(atms.n_atoms):
        atm_i = atms[i]

        for j in range(atms.n_atoms):
            if zero_vec and i == j:
                continue
            atm_j = atms[j]
            
            r_ij = np.linalg.norm(atm_i.position - atm_j.position + n_vec*L)

            e_pot += coul_fn(atm_i.charge, atm_j.charge, r_ij, **kwargs)
    #embed()
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

univ = MDAnalysis.Universe('run.tpr', 'traj.xtc')
atms = univ.atoms
n_atoms = atms.n_atoms

# for determining sigma...
cutoff = 1e-5

# alpha for screening charge
screen_alpha = get_alpha_from_cutoff(cutoff)

# alpha for test charge
test_alpha = 0.1

box = univ.dimensions[:3]
assert box[0] == box[1] == box[2]
L = box[0]


univ.trajectory[i_frame]


# Direct sum
max_vec = 0

# shift vectors
n = [np.array(n_vec) for n_vec in itertools.product(range(-max_vec, max_vec+1), repeat=3)]

pot_dir = 0
for n_vec in n:
    pot_dir += get_electro_dir(atms, n_vec, L, coul_sr, alpha=screen_alpha)
    #embed()
    
pot_dir = 0.5*k_e*pot_dir


# SR potential for (unscreened) gaussians

pot_gaus = 0
for n_vec in n:
    pot_gaus += get_electro_dir(atms, n_vec, L, coul_erf, alpha1=test_alpha)

pot_gaus = 0.5*k_e*pot_gaus

# Recip sum
max_vec = 12

m = [np.array(m_vec) for m_vec in itertools.product(range(-max_vec, max_vec+1), repeat=3)]

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

pot_self = k_e * (1/(np.sqrt(2*np.pi)*sig)) * np.sum([atm.charge**2 for atm in atms])

pot_recip -= pot_self

