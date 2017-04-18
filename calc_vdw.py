## Do vdw calculation from md trajectory##

from __future__ import division, print_function

import MDAnalysis
import numpy as np

import argparse

univ = MDAnalysis.Universe('top_noindus.tpr', '../stateC_bulk.gro')
univ.atoms.positions = univ.atoms.positions / 10.0

# (zero-indexed) indices of atoms that 
#    Will be alchemically transformed
#  For purposes of getting delta U, 
#  We need only consider paired VdW
#  interactions with these atoms and 
#  all other atoms
alc_indices = np.arange(877, 888)

atm_indices = np.arange(univ.atoms.n_atoms)

# set combined sigma to this if it's smaller
sc_sigma = 0.3

excls = {
         877:(870, 872, 873, 874, 875, 876, 877, 878, 
            879, 880, 884, 888),
         878:(870, 872, 873, 874, 875, 876, 877, 878, 
            879, 880, 881, 882, 883, 884, 885, 886, 887, 888),
         879:(872, 874, 875, 876, 877, 878, 879, 880, 
            881, 882, 883, 884, 885, 886, 887),
         880:(872, 874, 875, 876, 877, 878, 879, 880, 
            881, 882, 883, 884, 885, 886, 887),
         881:(874, 878, 879, 880, 881, 882, 883, 884),
         882:(874, 878, 879, 880, 881, 882, 883, 884),
         883:(874, 878, 879, 880, 881, 882, 883, 884),
         884:(872, 874, 875, 876, 877, 878, 879, 880, 
            881, 882, 883, 884, 885, 886, 887),
         885:(874, 878, 879, 880, 884, 885, 886, 887),
         886:(874, 878, 879, 880, 884, 885, 886, 887),
         887:(874, 878, 879, 880, 884, 885, 886, 887)
}

type_lookup = {
    'N3': 0,
    'H': 1,
    'CT': 2,
    'HP': 3,
    'HC': 4,
    'C': 5,
    'O': 6,
    'N': 7,
    'H1': 8,
    'SH': 9,
    'HS': 10,
    'OH': 11,
    'HO': 12,
    'CA': 13,
    'HA': 14,
    'O2': 15,
    'CC': 5,
    'NB': 16,
    'CR': 17,
    'H5': 18,
    'NA': 7,
    'CW': 17,
    'H4': 19,
    'DUM_HC': 20,
    'DUM_CT': 21,
    'DUM': 22,
    'OW_spc': 23,
    'HW_spc': 20
}
#format: 24 atom types, atomtype i is a tuple with (sigma, eps)
atmtypes = [
(3.25000e-01,  7.11280e-01), # N3
(1.06908e-01,  6.56888e-02), # H
(3.39967e-01,  4.57730e-01), # CT
(1.95998e-01,  6.56888e-02), # HP
(2.64953e-01,  6.56888e-02), # HC
(3.39967e-01,  3.59824e-01), # C
(2.95992e-01,  8.78640e-01), # O
(3.25000e-01,  7.11280e-01), # N
(2.47135e-01,  6.56888e-02), # H1
(3.56359e-01,  1.04600e+00), # SH
(1.06908e-01,  6.56888e-02), # HS
(3.06647e-01,  8.80314e-01), # OH
(0.00000e+00,  0.00000e+00), # HO
(3.39967e-01,  3.59824e-01), # CA
(2.59964e-01,  6.27600e-02), # HA
(2.95992e-01,  8.78640e-01), # O2
(3.25000e-01,  7.11280e-01), # NB
(3.39967e-01,  3.59824e-01), # CR
(2.42146e-01,  6.27600e-02), # H5
(2.51055e-01,  6.27600e-02), # H4
(0.00000e+00,  0.00000e+00), # DUM_HC
(0.00000e+00,  0.00000e+00), # DUM_CT
(0.00000e+00,  0.00000e+00), # DUM
(3.16557e-01,  6.50629e-01)  # OW_spc
]

n_atmtype = len(atmtypes)
# Generate VdW lookup table
c6_lut = np.zeros(n_atmtype**2)
c12_lut = np.zeros(n_atmtype**2)
sig_lut = np.zeros(n_atmtype**2)
sig6_lut = np.zeros(n_atmtype**2)
for i, payload_i in enumerate(atmtypes):
    sig_i, eps_i = payload_i
    for j, payload_j in enumerate(atmtypes):
        sig_j, eps_j = payload_j

        idx = i*n_atmtype + j
        sig = (sig_i + sig_j) / 2.0
        if sig < sc_sigma:
            sig = sc_sigma

        eps = np.sqrt(eps_i*eps_j)

        sig_lut[idx] = sig
        sig6_lut[idx] = sig**6
        c6_lut[idx] = 4*eps*sig**6
        c12_lut[idx] = 4*eps*sig**12

lmbda = 1.0
lmbda_lo = lmbda-0.1
lmbda_hi = lmbda+0.1

alpha = 0.5
u_lmbda = 0.0
u_lo = 0.0
u_hi = 0.0
# Calculate VdW energy differences between lambdas
for i in alc_indices[1:]:
    incl_indices = np.setdiff1d(atm_indices, excls[i])
    atm_i = univ.atoms[i]
    type_i = type_lookup[atm_i.type]

    for j in incl_indices:
        atm_j = univ.atoms[j]
        type_j = type_lookup[atm_j.type]

        lut_idx = type_i * n_atmtype + type_j

        r_ij_sq = np.sum((atm_i.position - atm_j.position)**2)

        c6 = c6_lut[lut_idx]
        c12 = c12_lut[lut_idx]
        sig = sig_lut[lut_idx]
        sig6 = sig6_lut[lut_idx]

        denom_lmbda = (alpha*sig6*lmbda + r_ij_sq**3)
        denom_lo = (alpha*sig6*lmbda_lo + r_ij_sq**3)
        denom_hi = (alpha*sig6*lmbda_hi + r_ij_sq**3)

        u_lmbda += (1-lmbda) * (c12/denom_lmbda**2 - c6/denom_lmbda)
        u_lo += (1-lmbda_lo) * (c12/denom_lo**2 - c6/denom_lo)
        u_hi += (1-lmbda_hi) * (c12/denom_hi**2 - c6/denom_hi)


