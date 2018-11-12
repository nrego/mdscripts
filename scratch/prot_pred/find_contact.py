from __future__ import division, print_function

import MDAnalysis
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from IPython import embed
from mdtools import MDSystem
import cPickle as pickle
import argparse

parser = argparse.ArgumentParser('Find buried atoms, surface atoms (and mask), and dewetted atoms')
parser.add_argument('-s', '--topology', type=str, required=True,
                    help='Input topology file')
parser.add_argument('-c', '--struct', type=str, required=True,
                    help='Input structure file')
parser.add_argument('--ref', type=str, required=True,
                    help='rho_data_dump of the reference structure')
parser.add_argument('--rhodata', type=str, required=True, 
                    help='rho_data_dump file for which to find dewetted atoms')
parser.add_argument('-nb', default=5, type=float,
                    help='Solvent exposure criterion for determining buried atoms from reference')
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Dewetting threshold for normalized rho (default: 0.5)')
parser.add_argument('--sel-spec', default='segid targ', type=str,
                    help='Selection spec for getting protein atoms')
parser.add_argument('--min-dist', type=str,
                    help='Optional: List of min dist of each of this selections atoms to the binding partner')
parser.add_argument('--actual-contact', type=str,
                    help='Optional: supply a mask of the actual contacts, for comparison')
parser.add_argument('--r-dist', type=float, default=4.0,
                    help='If min-dist is supplied, then only atoms this close will be considered contacts')


args = parser.parse_args()

sys = MDSystem(args.topology, args.struct, sel_spec=args.sel_spec)

ref_data = np.load(args.ref)['rho_water'].mean(axis=0)
targ_data = np.load(args.rhodata)['rho_water'].mean(axis=0)

sys.find_buried(ref_data, nb=args.nb)

# Surface heavy atoms
surf_mask = sys.surf_mask_h
buried_mask = ~surf_mask
np.savetxt('buried_mask.dat', buried_mask, fmt='%1d')
prot = sys.prot_h

rho_i = targ_data / ref_data

# Are we predicting contacts or finding them from a bound simulation?
if args.min_dist is None:
    print('predicting contacts from phi-ens simulation...')
    contact_mask = (rho_i < args.thresh) & surf_mask
    prot[contact_mask].tempfactors = 1

    np.savetxt('pred_contact_mask.dat', contact_mask, fmt='%1d')

    prot.write('pred_contact.pdb', bonds=None)

    prot.tempfactors = rho_i
    prot[buried_mask].tempfactors = -2
    prot.write('pred_contact_rho.pdb', bonds=None)

    if args.actual_contact is not None:
        print('...and comparing to actual contacts')

        actual_contact_mask = np.loadtxt(args.actual_contact).astype(bool)[surf_mask]
        pred_contact_mask = (rho_i < args.thresh)[surf_mask]

        tp = pred_contact_mask & actual_contact_mask
        fp = pred_contact_mask & ~actual_contact_mask
        tn = ~pred_contact_mask & ~actual_contact_mask
        fn = ~pred_contact_mask & actual_contact_mask

        prot.tempfactors = -2
        prot[surf_mask][tp].tempfactors = 1
        prot[surf_mask][fp].tempfactors = -1
        prot.write('pred_contact_tp_fp.pdb', bonds=None)

        np.savetxt('accuracy.dat', np.array([tp.sum(), fp.sum(), tn.sum(), fn.sum()]), fmt='%.1f')

else:
    print('Finding contacts from bound simulation...')
    min_dist = np.load(args.min_dist)['min_dist'].mean(axis=0)

    contact_mask = (min_dist < args.r_dist) & surf_mask
    contact_mask_dewet = contact_mask & (rho_i < args.thresh)
    prot[contact_mask].tempfactors = 1

    np.savetxt('actual_contact_mask.dat', contact_mask, fmt='%1d')
    np.savetxt('actual_contact_mask_phob.dat', contact_mask_dewet, fmt='%1d')
    prot.write('actual_contact.pdb', bonds=None)

    prot.tempfactors = rho_i
    prot[~contact_mask].tempfactors = -2
    prot.write('actual_contact_rho.pdb', bonds=None)


sys.other.write('other.pdb', bonds=None)
