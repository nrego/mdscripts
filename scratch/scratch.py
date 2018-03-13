from __future__ import division, print_function

import numpy
from matplotlib import pyplot
import argparse
import logging
from datareader import DataReader as dr

from whamutils import gen_U_nm, kappa, grad_kappa, gen_pdist

import numpy as np
from scipy.optimize import fmin_l_bfgs_b as fmin_bfgs
import pymbar

from IPython import embed


fermi = lambda x : 1 / (1 + np.exp(x))

def iter_wham(d0, d1, c):

    sum0 = fermi(d0-c).mean()
    sum1 = fermi(-d1+c).mean()

    return np.log(sum1) - np.log(sum0) + c #- np.log(n_sample1/n_sample0)

def converge_wham(d0, d1, c, n_iter=1000):
    last_c = c
    dg = 0

    n0 = d0.size
    n1 = d1.size
    n_out = n_iter / 10

    for i in range(n_iter):

        if i % n_out == 0:
            print("dg={}".format(dg))
            print("C={}".format(c))

        dg = iter_wham(d0, d1, c)

        c = dg + np.log(n1/n0)

    return dg

ds0 = dr.loadXVG('lambda_025/dhdl.xvg')
ds1 = dr.loadXVG('lambda_050/dhdl.xvg')

lam0 = ds0.lmbda
lam1 = ds1.lmbda

k = 0.0083144598
start = 2000

## Extract Delta U's in reduced units ##
beta = 1/(k*300)

dat0 =  beta*np.array(ds0.data[start:][lam1])
dat1 = -beta*np.array(ds1.data[start:][lam0])

## Make histograms of P(Delta U) for overlap comparison ##
minpt = min(dat0.min(), dat1.min())
maxpt = max(dat0.max(), dat1.max())

binbounds = np.arange(minpt, maxpt, 0.01)

hist0, bb = np.histogram(dat0, bins=binbounds)
hist1, bb = np.histogram(dat1, bins=binbounds)

bc = binbounds[:-1] + np.diff(binbounds)/2.0

## Get IACTs

iact0 = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dat0, fast=True))
iact1 = np.ceil(pymbar.timeseries.integratedAutocorrelationTime(dat1, fast=True))

# block sizes separating independent samples #
block0 = int(1 + 2*iact0)
block1 = int(1 + 2*iact1)

autocorr = np.array([block0, block1])

## Perform analysis on entire dataset and uncorrelated datasets

n_samples = np.array([dat0.size, dat1.size])
uncorr_n_samples = n_samples // autocorr
remainders = n_samples % autocorr

uncorr_dat0 = dat0[remainders[0]::block0]
uncorr_dat1 = dat1[remainders[1]::block1]

assert uncorr_n_samples[0] == uncorr_dat0.size
assert uncorr_n_samples[1] == uncorr_dat1.size

n_tot = n_samples.sum()
uncorr_n_tot = uncorr_n_samples.sum()

bias_mat = np.zeros((n_tot, 2), dtype=np.float64)
uncorr_bias_mat = np.zeros((uncorr_n_tot, 2), dtype=np.float64)

bias_mat[:n_samples[0], 1] = dat0
bias_mat[n_samples[0]:, 1] = dat1
uncorr_bias_mat[:uncorr_n_samples[0], 1] = uncorr_dat0
uncorr_bias_mat[uncorr_n_samples[0]:, 1] = uncorr_dat1

n_sample_diag = np.matrix(np.diag(n_samples / n_tot), dtype=np.float32)
ones_m = np.matrix(np.ones(2,), dtype=np.float32).T
ones_n = np.matrix(np.ones(n_tot,), dtype=np.float32).T

uncorr_n_sample_diag = np.matrix(np.diag(uncorr_n_samples / uncorr_n_tot), dtype=np.float32)
uncorr_ones_m = np.matrix(np.ones(2,), dtype=np.float32).T
uncorr_ones_n = np.matrix(np.ones(uncorr_n_tot,), dtype=np.float32).T

myargs = (bias_mat, n_sample_diag, ones_m, ones_n, n_tot)
xweights = np.array([0.])

#fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=myargs)[0]

uncorr_myargs = (uncorr_bias_mat, uncorr_n_sample_diag, uncorr_ones_m, uncorr_ones_n, uncorr_n_tot)
xweights = np.array([0.])

fmin_bfgs(kappa, xweights, fprime=grad_kappa, args=uncorr_myargs)[0]