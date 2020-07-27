
import numpy as np

import argparse
import logging

import scipy

import os, glob

from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from constants import k
from IPython import embed
import math

import sys

from whamutils import get_negloghist, extract_and_reweight_data

import argparse

from work_managers.environment import default_env


## LOAD RHO (x,y,z) density profiles
#
#. Reweight to given ensemble (at different bphi vals, at different N vals)

def load_and_weight(idx, fnames, logweights):

    logweights -= logweights.max()
    norm = np.log(np.sum(np.exp(logweights)))
    logweights -= norm
    weights = np.exp(logweights)

    assert np.allclose(weights.sum(), 1)

    xbins = None
    ybins = None
    zbins = None

    nx = None
    ny = None
    nz = None

    last_frame = 0

    all_rhoxzy = []

    for fname in fnames:
        #print(fname)
        ds = np.load(fname)

        if xbins is None:
            xbins = ds['xbins']
            nx = xbins.size - 1
        if ybins is None:
            ybins = ds['ybins']
            ny = ybins.size - 1
        if zbins is None:
            zbins = ds['zbins']
            nz = zbins.size - 1

        assert np.array_equal(ds['xbins'], xbins)
        assert np.array_equal(ds['ybins'], ybins)
        assert np.array_equal(ds['zbins'], zbins)

        this_rhoxyz = ds['rho']
        assert this_rhoxyz.shape[1:] == (nx, ny, nz)

        this_frames = this_rhoxyz.shape[0]
        this_weights = weights[last_frame:last_frame+this_frames]

        this_weight_rho = np.dot(this_weights, this_rhoxyz.reshape(this_frames, -1)).reshape(nx, ny, nz)

        all_rhoxzy.append(this_weight_rho)
        

        last_frame += this_frames

        del this_rhoxyz, this_weights, this_weight_rho

    all_rhoxzy = np.array(all_rhoxzy)


    return idx, all_rhoxzy.sum(axis=0)


### SET THIS TO TASTE AROUND BPHI *
## LOAD IN N v Phi

parser = argparse.ArgumentParser('Reweight density fields rho(x,y,z) to ensembles around bphistar and constrained on vals of n')
parser.add_argument('--nvphi', type=str, default='../NvPhi.dat', help='location of n v phi data')
parser.add_argument('--smooth-width', type=int, default=5,
                    help='Smoothing width for n reweighting (+- smooth on each side for each n)')

default_env.add_wm_args(parser)

args = parser.parse_args()
default_env.process_wm_args(args)
wm = default_env.make_work_manager()
wm.startup()


print("READING N v Phi DATA FROM: {}".format(args.nvphi))
dat = np.loadtxt(args.nvphi)
n_buffer = args.smooth_width 

max_idx = np.argmax(dat[:,2])
bphistar = dat[max_idx, 0]
avg_n_bphistar = dat[max_idx, 1]
sus_bphistar = dat[max_idx, 2]

low_mask = dat[:,2] < 0.5*sus_bphistar

## First or last indices where chi is less than 0.5*chimax
phiminus_idx = (~low_mask).argmax() - 1
phiplus_idx = low_mask.size - (~low_mask[::-1]).argmax()

bphi_minus = dat[phiminus_idx, 0]
bphi_plus = dat[phiplus_idx, 0]
avg_n_minus = dat[phiminus_idx, 1]
avg_n_plus = dat[phiplus_idx, 1]

print("\nbphistar: {:.2f} <N_v>bphistar: {:.2f}".format(bphistar, avg_n_bphistar))
print("bphi(-): {:.2f} <N_v>bphi(-): {:.2f}".format(bphi_minus, avg_n_minus))
print("bphi(+): {:.2f} <N_v>bphi(+): {:.2f}\n".format(bphi_plus, avg_n_plus))

## Set to for N values around <N_v>bphistar
nvals = np.arange(np.ceil(avg_n_minus), np.floor(avg_n_plus)-1, -1)
beta_phi_vals = np.arange(bphi_minus, bphi_plus+0.02, 0.02)

print("nvals: {}".format(nvals))
print("beta phi vals: {}".format(beta_phi_vals))

sys.stdout.flush()

#### DONE READING N V PHI ###


### EXTRACT MBAR DATA ###

print("\nREADING MBAR DATA...")

all_data_ds = np.load('all_data.dat.npz')
all_logweights = all_data_ds['logweights']
all_data = all_data_ds['data']
all_data_N = all_data_ds['data_aux']

print("...done...\n")
## If 2d density profiles are ready - reweight them, too, to find the unbiased rho(z,r) ###

# Now check to see if rho_z data is ready

fnames_rhoxyz = sorted(glob.glob("*/rhoxyz.dat.npz"))

if len(fnames_rhoxyz) > 0:
    print("Doing rhoxyz...\n")
else:
    print("Done. Goodbye.")
    sys.exit()

ds = np.load(fnames_rhoxyz[0])


### CALCULATE RHO(x,y,z) with differnt BPHI vals
################################################


### FIRST UNBIASED ####
print("\nCalculating rho, (equil)...\n")
sys.stdout.flush()
_, rho0 = load_and_weight(-1, fnames_rhoxyz, all_logweights)
print("...done\n")


## Next, rho(x,y,z) for each of the beta phi values...
print("\nCalculating rho for different bphi values...\n")
sys.stdout.flush()


rho_bphi = np.zeros((beta_phi_vals.size, *rho0.shape))

# Generator for reweighting w/ bphi
def task_gen_bphi():
    
    for i, beta_phi_val in enumerate(beta_phi_vals):
        #print("  doing {} of {}\n".format(i, beta_phi_vals.size))

        bias_logweights = all_logweights - beta_phi_val*all_data

        args = (i, fnames_rhoxyz, bias_logweights)
        kwargs = dict()
        print("sending bphi {:.2f} ({} of {})".format(beta_phi_val, i, beta_phi_vals.size))


        yield load_and_weight, args, kwargs


for future in wm.submit_as_completed(task_gen_bphi(), queue_size=wm.n_workers):
    idx, rho = future.get_result(discard=True)
    print("receiving result...")
    sys.stdout.flush()

    rho_bphi[idx] = rho

    del rho

print("...done\n")

### CALCULATE RHO(x,y,z) with different n vals
################################################
print("\nCalculating rho for different n vals (smooth width: {})".format(n_buffer))
sys.stdout.flush()

# rho(x,y,z) with n
rho_n = np.zeros((nvals.size, *rho0.shape))

def task_gen_nvals():
    
    for i, nval in enumerate(nvals):
        #print("  doing {} of {}\n".format(i, beta_phi_vals.size))
        mask = (all_data_N >= nval-n_buffer) & (all_data_N < nval+n_buffer)

        bias_logweights = np.zeros_like(all_logweights)
        bias_logweights[:] = -np.inf
        bias_logweights[mask] = all_logweights[mask]

        args = (i, fnames_rhoxyz, bias_logweights)
        kwargs = dict()
        print("sending nval {} ({} of {})".format(nval, i, nvals.size))


        yield load_and_weight, args, kwargs


for future in wm.submit_as_completed(task_gen_bphi(), queue_size=wm.n_workers):
    idx, rho = future.get_result(discard=True)
    print("receiving result...")
    sys.stdout.flush()

    rho_n[idx] = rho

    del rho

print("Finished, saving...")

np.savez_compressed('rho_final.dat', rho_bphi=rho_bphi, beta_phi_vals=beta_phi_vals, rho_n=rho_n, nvals=nvals,
                    rho0=rho0, xbins=ds['xbins'], ybins=ds['ybins'], zbins=ds['zbins'])

