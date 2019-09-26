## My set of WHAM/MBAR utilities
from __future__ import division

import numpy as np

from IPython import embed
import numexpr

import sys

import math

#from IPython import embed
# Generate a probability distribution over a variable by integrating
# This currently works for phi datasets **ONLY**
def gen_pdist(all_dat, bias_mat, n_samples, f_k, binbounds):
    #bias_mat = np.exp(-bias_mat)

    weights_prod_nsample = n_samples * np.exp(f_k)

    pdist = np.zeros(binbounds.shape[0]-1, dtype=np.float64)

    for n_idx in range(all_dat.shape[0]):
        denom_arr = f_k - bias_mat[n_idx, :]
        denom_arr = n_samples * np.exp(denom_arr)
        denom = denom_arr.sum()
        if all_dat.ndim == 2:
            for k in range(all_dat.shape[1]):
                val = all_dat[n_idx, k]
                bin_assign = (val >= binbounds[:-1]) * (val < binbounds[1:])
                #denom = np.dot(weights_prod_nsample, bias_mat[n_idx, :])
                pdist[bin_assign] += 1.0/denom

        elif all_dat.ndim == 1:
            val = all_dat[n_idx]

            # which bin does ntwid fall into? A boolean array.
            # Might want to assert sum==1 here; i.e. it falls into exactly 1 bin
            bin_assign = (val >= binbounds[:-1]) * (val < binbounds[1:])

            #denom = np.dot(weights_prod_nsample, bias_mat[n_idx, :])

            pdist[bin_assign] += 1.0/denom

    return pdist

# Generate weights for each datapoint (from a wham simulation)
#   according to a bias matrix (the biases over all windows, applied to all datapoints)
#   and the generated f_k's (from WHAM, the free energy of applying each window's bias)
#
#   Returns:   logweights  (n_tot, ) [weight for each datapoint]
#
# If dealing with full dataset w/ stat inefficiences, ones_m and ones_N must contain them
#   If dealing with subsampled (uncorrelated) data, ones_m and ones_N are just arrays of ones,
#      and n_samples is just the number of uncorrelated samples for each window
def gen_data_logweights(bias_mat, f, n_samples, ones_m, ones_N):

    Q = f - bias_mat
    denom = np.log(n_samples * ones_m) + Q
    c = denom.max(axis=1)
    denom -= c[:,None]

    denom = np.log(np.sum(np.exp(denom), axis=1)) + c
    numer = np.log(ones_N)

    logweights = numer - denom

    logweights -= logweights.max()
    norm = np.log(math.fsum(np.exp(logweights)))
    logweights -= norm

    return logweights


# Log likelihood
def kappa(xweights, bias_mat, nsample_diag, ones_m, ones_N, n_tot):
    
    f = np.append(0, xweights)
    
    # n_tot x n_w
    Q = f - bias_mat

    # n_w x 1
    diag = np.log(np.dot(nsample_diag, ones_m)).T
    P = Q + diag
    c = P.max(axis=1)
    P -= c[:,None]

    #ln_sum_exp = np.log(np.exp(P).sum(axis=1)) + c
    ln_sum_exp = np.log(numexpr.evaluate("exp(P)").sum(axis=1)) + c

    logLikelihood = np.dot((ones_N.T/n_tot), ln_sum_exp) - \
                    np.dot(np.dot(ones_m.T, nsample_diag), f)

    return logLikelihood.item()

def grad_kappa(xweights, bias_mat, nsample_diag, ones_m, ones_N, n_tot):

    #u_nm = np.exp(-bias_mat)
    f = np.append(0, xweights)

    # n_tot x n_w
    Q = f - bias_mat

    # n_w x 1
    diag = np.log(np.dot(nsample_diag, ones_m)).T
    c = Q.max(axis=1)
    Q -= c[:,None]
    P = Q + diag

    #denom = np.exp(P).sum(axis=1)[:,None] #+ c
    denom = numexpr.evaluate("exp(P)").sum(axis=1)[:,None] #+ c

    #W = np.exp(Q)/denom
    W = numexpr.evaluate("exp(Q)")/denom
    
    grad = np.dot(np.dot(nsample_diag,W.T), (ones_N/n_tot)) * ones_m - np.dot(nsample_diag,ones_m)


    return ( grad[1:] ).squeeze()

def hess_kappa(xweights, bias_mat, nsample_diag, ones_m, ones_N, n_tot):
    
    f = np.append(0, xweights)

    # n_tot x n_w
    Q = f - bias_mat

    # n_w x 1
    diag = np.log(np.dot(nsample_diag, ones_m)).T
    c = Q.max(axis=1)
    Q -= c[:,None]
    P = Q + diag

    #denom = np.exp(P).sum(axis=1)[:,None] #+ c
    denom = numexpr.evaluate("exp(P)").sum(axis=1)[:,None] #+ c

    #W = np.exp(Q)/denom
    W = numexpr.evaluate("exp(Q)")/denom
    #embed()
    new_diag = np.dot(np.diag(ones_m), nsample_diag)
    hess = -np.dot(np.dot(new_diag, W.T), np.dot(W, new_diag.T)) * np.dot(ones_m[:,None], ones_m[None,:]) / n_tot

    return hess[1:,1:]

Nfeval = 0
def callbackF(xweights):
    global Nfeval
    #log.info('Iteration {}'.format(Nfeval))
    print('Iteration {}  weights: {:.2f}'.format(Nfeval, xweights[0]), end='\r')
    sys.stdout.flush()
    #log.info('.')

    Nfeval += 1