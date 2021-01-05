
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


# IDX: index of filename
# logweights consists of *only* those weights associated with datapoints from fname
#@profile
def load_and_weight_file(idx, fname, logweights, xbins, ybins, zbins, hat_cav):
    #print(fname)

    nx = xbins.size - 1
    ny = ybins.size - 1
    nz = zbins.size - 1

    ## No weight for this sample; return zero density array
    if logweights.max() == -np.inf:
        return idx, np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))

    ## Non-zero weight associated with this fname; normalize...
    weights = np.exp(logweights)

    with np.load(fname) as ds:

        assert np.array_equal(ds['xbins'], xbins)
        assert np.array_equal(ds['ybins'], ybins)
        assert np.array_equal(ds['zbins'], zbins)

        this_cav = ds['cav'].astype(int)

    assert this_cav.shape[1:] == (nx, ny, nz)
    assert weights.size == this_cav.shape[0]

    this_cav_shape = this_cav.reshape(this_cav.shape[0], -1)
    
    avg_cav = np.dot(weights, this_cav_shape)
    avg_cav_sq = np.dot(weights, this_cav_shape**2)

    del logweights, this_cav, this_cav_shape, weights, ds


    return idx, avg_cav.reshape(nx,ny,nz), avg_cav_sq.reshape(nx,ny,nz)

def find_hat_cav_diff(idx, fname, logweights, xbins, ybins, zbins, hat_cav):
    #print(fname)

    nx = xbins.size - 1
    ny = ybins.size - 1
    nz = zbins.size - 1

    ## No weight for this sample; return zero density array
    if logweights.max() == -np.inf:
        return idx, np.zeros((nx, ny, nz))

    ## Non-zero weight associated with this fname; normalize...
    weights = np.exp(logweights)

    with np.load(fname) as ds:

        assert np.array_equal(ds['xbins'], xbins)
        assert np.array_equal(ds['ybins'], ybins)
        assert np.array_equal(ds['zbins'], zbins)

        this_cav = ds['cav'].astype(int)

    assert this_cav.shape[1:] == (nx, ny, nz)
    assert weights.size == this_cav.shape[0]

    this_cav_shape = this_cav.reshape(this_cav.shape[0], -1)
    hat_cav_shape = hat_cav.ravel()

    diff = this_cav_shape - hat_cav_shape
    
    avg_diff = np.dot(weights, diff)

    del logweights, this_cav, this_cav_shape, hat_cav_shape, weights, ds


    return idx, avg_diff.reshape(nx,ny,nz)


### SET THIS TO TASTE AROUND BPHI *
## LOAD IN N v Phi

parser = argparse.ArgumentParser('Reweight density fields rho(x,y,z) to ensembles around bphistar and constrained on vals of n')
parser.add_argument('--nvphi', type=str, default='../NvPhi.dat', help='location of n v phi data')
parser.add_argument('--smooth-width', type=int, default=5,
                    help='Smoothing width for n reweighting (+- smooth on each side for each n)')
parser.add_argument('--dry-run', action='store_true', help='if true, just print out range of nvals and bphis')
parser.add_argument('--do-rho0', action='store_true', help='if true, calculate equil density field')
parser.add_argument('--n-val', type=int, help='calculate rhoxyz density field at this particular value of N')
parser.add_argument('--bphi-val', type=float, help='calculate rhoxyz density field at this particular value of bphi')

default_env.add_wm_args(parser)

args = parser.parse_args()
default_env.process_wm_args(args)
wm = default_env.make_work_manager()
#wm.startup()


print("READING N v Phi DATA FROM: {}".format(args.nvphi))
dat = np.loadtxt(args.nvphi)
n_buffer = args.smooth_width 

# phistar
max_idx = np.argmax(dat[:,2])
bphistar = dat[max_idx, 0]
avg_n_bphistar = dat[max_idx, 1]
sus_bphistar = dat[max_idx, 2]

low_mask = dat[:,2] < 0.5*sus_bphistar


## Find phi(-) and phi(+), as well as average N's for this range
######
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
nvals = np.arange(np.ceil(avg_n_minus), -1, -1)
beta_phi_vals = np.arange(bphi_minus, bphi_plus+0.02, 0.02)

print("nvals: {}".format(nvals))
print("beta phi vals: {}".format(beta_phi_vals))

sys.stdout.flush()

if args.dry_run:
    sys.exit()

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


print("\nDetermining number of points for each file...")
n_frames_per_file = np.zeros(len(fnames_rhoxyz), dtype=int)

xbins = None
ybins = None
zbins = None

for i, fname in enumerate(fnames_rhoxyz):
    print("checking {}".format(fname))
    with np.load(fname) as ds:
        n_frames_per_file[i] = ds['nframes']

        if xbins is None:
            xbins = ds['xbins']
            nx = xbins.size-1
        if ybins is None:
            ybins = ds['ybins']
            ny = ybins.size-1
        if zbins is None:
            zbins = ds['zbins']
            nz = zbins.size-1

        assert np.array_equal(xbins, ds['xbins'])
        assert np.array_equal(ybins, ds['ybins'])
        assert np.array_equal(zbins, ds['zbins'])

        #cav = ds['cav']
print("\n...Done.\n")


### CALCULATE RHO(x,y,z) with different BPHI vals, N, etc...
################################################

## Generator/task distributor
def task_gen(fn, fnames, n_frames_per_file, logweights, xbins, ybins, zbins, hat_cav=None):

    # Normalize logweights
    # -= logweights.max()
    #norm = np.log(np.exp(logweights).sum())
    #logweights -= norm

    last_frame = 0

    ## For each rhoxyz.dat.npz...
    for i, fname in enumerate(fnames):
        
        print("sending {:04d} {}".format(i, fname))
        # Slice of logweights corresponding to this rhoxyz cavity profile
        slc = slice(last_frame, last_frame+n_frames_per_file[i])
        this_logweights = logweights[slc]
        
        args = (i, fname, this_logweights, xbins, ybins, zbins, hat_cav)
        kwargs = dict()
        last_frame += n_frames_per_file[i]

        yield fn, args, kwargs


if args.bphi_val is not None:

    beta_phi_val = args.bphi_val
    print("\nCalculating rho for bphi={:.4f}...\n".format(beta_phi_val))
    sys.stdout.flush()


    bias_logweights = all_logweights - beta_phi_val * all_data
    bias_logweights -= bias_logweights.max()
    norm = np.log(np.sum(np.exp(bias_logweights)))
    bias_logweights -= norm

    assert np.allclose(1, np.exp(bias_logweights).sum())

    avg_cav = np.zeros((nx, ny, nz))
    avg_cav_sq = np.zeros((nx, ny, nz))
    
    with wm:
        for future in wm.submit_as_completed(task_gen(load_and_weight_file, fnames_rhoxyz, n_frames_per_file, bias_logweights, xbins, ybins, zbins), queue_size=wm.n_workers):
            idx, this_avg_cav, this_avg_cav_sq = future.get_result(discard=True)
            avg_cav += this_avg_cav
            avg_cav_sq += this_avg_cav_sq

    # binary density field from avg_cav
    hat_cav = avg_cav < 0.5
    mse_hat_cav = np.zeros((nx, ny, nz))

    print("...done\n")
    print("Next up: calculating devation from cg density field...\n")
    
    with wm:
        for future in wm.submit_as_completed(task_gen(find_hat_cav_diff, fnames_rhoxyz, n_frames_per_file, bias_logweights, xbins, ybins, zbins, hat_cav), queue_size=wm.n_workers):
            idx, this_hat_cav_diff = future.get_result(discard=True)
            mse_hat_cav += this_hat_cav_diff    

    print("...done\n")

    np.savez_compressed('rho_bphi_{:.4f}.dat'.format(beta_phi_val), avg_cav=avg_cav, avg_cav_sq=avg_cav_sq, beta_phi_val=beta_phi_val,
                        xbins=xbins, ybins=ybins, zbins=zbins)

if args.n_val is not None:

    n_val = args.n_val
    print("\nCalculating rho for n={:04d}...\n".format(n_val))
    sys.stdout.flush()

    mask = (all_data_N >= n_val-n_buffer) & (all_data_N <= n_val+n_buffer)
    bias_logweights = np.zeros_like(all_logweights)
    bias_logweights[:] = -np.inf
    bias_logweights[mask] = all_logweights[mask]
    bias_logweights -= bias_logweights.max()
    norm = np.log(np.sum(np.exp(bias_logweights)))
    bias_logweights -= norm

    assert np.allclose(1, np.exp(bias_logweights).sum())

    avg_cav = np.zeros((nx, ny, nz))
    avg_cav_sq = np.zeros((nx, ny, nz))
    
    with wm:
        for future in wm.submit_as_completed(task_gen(load_and_weight_file, fnames_rhoxyz, n_frames_per_file, bias_logweights, xbins, ybins, zbins), queue_size=wm.n_workers):
            idx, this_avg_cav, this_avg_cav_sq = future.get_result(discard=True)
            avg_cav += this_avg_cav
            avg_cav_sq += this_avg_cav_sq

    print("...done\n")
    print("Next up: calculating devation from cg density field...\n")
    # binary density field from avg_cav
    hat_cav = avg_cav < 0.5
    mse_hat_cav = np.zeros((nx, ny, nz))

    with wm:
        for future in wm.submit_as_completed(task_gen(find_hat_cav_diff, fnames_rhoxyz, n_frames_per_file, bias_logweights, xbins, ybins, zbins, hat_cav), queue_size=wm.n_workers):
            idx, this_hat_cav_diff = future.get_result(discard=True)
            print("getting job: {:04d}".format(idx))
            mse_hat_cav += this_hat_cav_diff


    print("...done\n")

    np.savez_compressed('rho_n_{:04d}.dat'.format(n_val), avg_cav=avg_cav, avg_cav_sq=avg_cav_sq, n_val=n_val,
                        mse_hat_cav=mse_hat_cav, xbins=xbins, ybins=ybins, zbins=zbins)

