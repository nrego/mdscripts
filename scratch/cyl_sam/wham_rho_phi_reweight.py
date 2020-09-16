
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

#from memory_profiler import profile

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
        print(fname)
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


    return idx, all_rhoxzy

# IDX: index of filename
# logweights consists of *only* those weights associated with datapoints from fname
#@profile
def load_and_weight_file(idx, fname, logweights, nx, ny, nz, xbins, ybins, zbins):

    ## No weight for this sample; return zero density array
    if logweights.max() == -np.inf:
        return idx, np.zeros((nx, ny, nz))

    ## Non-zero weight associated with this fname; normalize...
    weights = np.exp(logweights)

    #print(fname)
    ds = np.load(fname)

    assert np.array_equal(ds['xbins'], xbins)
    assert np.array_equal(ds['ybins'], ybins)
    assert np.array_equal(ds['zbins'], zbins)


    this_rhoxyz = ds['rho']
    assert this_rhoxyz.shape[1:] == (nx, ny, nz)
    assert weights.size == this_rhoxyz.shape[0]

    this_weight_rho = np.dot(weights, this_rhoxyz.reshape(this_rhoxyz.shape[0], -1)).reshape(nx, ny, nz)

    ds.close()
    del logweights, this_rhoxyz, weights, ds


    return idx, this_weight_rho

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

if len(fnames_rhoxyz) > 0:
    print("Doing rhoxyz...\n")
else:
    print("Done. Goodbye.")
    sys.exit()

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

### CALCULATE RHO(x,y,z) with differnt BPHI vals, N, etc...
################################################

## Generator/task distributor
def task_gen(fnames, n_frames_per_file, logweights):

    last_frame = 0

    for i, fname in enumerate(fnames):
        
        #print(" {}".format(fname))
        slc = slice(last_frame, last_frame+n_frames_per_file[i])
        this_logweights = logweights[slc]
        
        args = (i, fname, this_logweights, nx, ny, nz, xbins, ybins, zbins)
        kwargs = dict()
        last_frame += n_frames_per_file[i]

        yield load_and_weight_file, args, kwargs

## Asynchronously launch reweighting's for each windows
def rho_job(rho_avg, wm, fnames_rhoxyz, n_frames_per_file, logweights):
    for future in wm.submit_as_completed(task_gen(fnames_rhoxyz, n_frames_per_file, logweights), queue_size=wm.n_workers):
        idx, rho = future.get_result(discard=True)
        print("  receiving result...")
        sys.stdout.flush()

        rho_avg += rho

        del rho

    return rho_avg

### FIRST: UNBIASED ####
if args.do_rho0:

    print("\nCalculating rho, (equil)...\n")
    sys.stdout.flush()
    
    logweights = all_logweights.copy()
    logweights -= logweights.max()
    norm = np.log(np.sum(np.exp(logweights)))
    logweights -= norm

    assert np.allclose(1, np.exp(logweights).sum())

    rho0 = np.zeros((nx, ny, nz))
    with wm:
        rho0 = rho_job(rho0, wm ,fnames_rhoxyz, n_frames_per_file, logweights)
    #sys.exit()

    print("...done\n")
    np.savez_compressed('rho0.dat', rho0=rho0, xbins=xbins, ybins=ybins, zbins=zbins)

    sys.exit()

## Now, rho(x,y,z) for each n val...

if args.n_val is not None:
    nval = args.n_val

    print("\nCalculating rho for n: {:04d}\n".format(n_val))
    sys.stdout.flush()

    rho_n = np.zeros((nx, ny, nz))

    #for i_nval, nval in enumerate(nvals):
    #print("doing n {}  ({} of {})".format(nval, i_nval+1, nvals.size))
    #sys.stdout.flush()

    mask = (all_data_N >= nval-n_buffer) & (all_data_N <= nval+n_buffer)
    bias_logweights = np.zeros_like(all_logweights)
    bias_logweights[:] = -np.inf
    bias_logweights[mask] = all_logweights[mask]
    bias_logweights -= bias_logweights.max()
    norm = np.log(np.sum(np.exp(bias_logweights)))
    bias_logweights -= norm

    assert np.allclose(1, np.exp(bias_logweights).sum())

    this_rho = np.zeros((nx, ny, nz))
    
    with wm:
        this_rho = rho_job(this_rho, wm, fnames_rhoxyz, n_frames_per_file, bias_logweights)


    print("Finished, saving...")
    sys.stdout.flush()


    np.savez_compressed('rho_n_{:04d}.dat'.format(n_val), rho_n=this_rho, nval=nval,
                        xbins=xbins, ybins=ybins, zbins=zbins)

    print("done.")
    sys.stdout.flush()

    sys.exit()

## Next, rho(x,y,z) for each of the beta phi values...

if args.bphi_val:

    beta_phi_val = args.bphi_val
    print("\nCalculating rho for bphi={:.4f}...\n".format(beta_phi_val))
    sys.stdout.flush()

    #rho_beta_phi = np.zeros((beta_phi_vals.size, nx, ny, nz))

    #for i_bphi, beta_phi_val in enumerate(beta_phi_vals):
        #print("doing bphi {:.2f}  ({} of {})".format(beta_phi_val, i_bphi+1, beta_phi_vals.size))
        #sys.stdout.flush()

    bias_logweights = all_logweights - beta_phi_val * all_data
    bias_logweights -= bias_logweights.max()
    norm = np.log(np.sum(np.exp(bias_logweights)))
    bias_logweights -= norm

    assert np.allclose(1, np.exp(bias_logweights).sum())

    this_rho = np.zeros((nx, ny, nz))
    
    with wm:
        this_rho = rho_job(this_rho, wm, fnames_rhoxyz, n_frames_per_file, bias_logweights)


    print("...done\n")

    np.savez_compressed('rho_bphi_{:.4f}.dat'.format(beta_phi_val), rho_bphi=this_rho, beta_phi_val=beta_phi_val,
                        xbins=xbins, ybins=ybins, zbins=zbins)


