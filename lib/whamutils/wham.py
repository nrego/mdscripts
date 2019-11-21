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


## Convenience methods to deal constructing prob distributions

# Get the (negative log) of the probability distribution
#   over a binned set of data with associated log weights
def get_negloghist(data, bins, logweights):

    logweights -= logweights.max()
    #norm = math.fsum(np.exp(logweights))
    norm = np.sum(np.exp(logweights))
    logweights -= np.log(norm)

    bin_assign = np.digitize(data, bins) - 1

    negloghist = np.zeros(bins.size-1)
    negloghist[:] = np.inf

    for i in range(bins.size-1):

        this_bin_mask = bin_assign == i
        if this_bin_mask.sum() == 0:
            continue

        this_logweights = logweights[this_bin_mask]
        this_data = data[this_bin_mask]

        this_logweights_max = this_logweights.max()
        this_logweights -= this_logweights.max()

        this_weights = np.exp(this_logweights)

        #negloghist[i] = -np.log(math.fsum(this_weights)) - this_logweights_max
        negloghist[i] = -np.log(np.sum(this_weights)) - this_logweights_max


    return negloghist


# Get PvN, Nvphi, and chi v phi for a set of datapoints and their weights
#  Note avg n v phi is *not* reweighted (i.e. it's under the phi*ntwid ensemble)
def extract_and_reweight_data(logweights, ntwid, data, bins, beta_phi_vals):
    
    neglogpdist = get_negloghist(ntwid, bins, logweights)
    neglogpdist_data = get_negloghist(data, bins, logweights)

    # Average Ntwid and var for each beta_phi
    avg_ntwid = np.zeros_like(beta_phi_vals)
    var_ntwid = np.zeros_like(avg_ntwid)

    # Same for data - also include covariance w/ ntwid
    avg_data = np.zeros_like(avg_ntwid)
    var_data = np.zeros_like(avg_ntwid)
    cov_data = np.zeros_like(avg_ntwid)

    ntwid_sq = ntwid**2
    data_sq = data**2

    cov = ntwid * data

    for i, beta_phi_val in enumerate(beta_phi_vals):

        bias_logweights = logweights - beta_phi_val*ntwid
        bias_logweights -= bias_logweights.max()
        #norm = np.log(math.fsum(np.exp(bias_logweights)))
        norm = np.log(np.sum(np.exp(bias_logweights)))
        bias_logweights -= norm

        bias_weights = np.exp(bias_logweights)

        this_avg_ntwid = np.dot(bias_weights, ntwid)
        this_avg_ntwid_sq = np.dot(bias_weights, ntwid_sq)
        this_var_ntwid = this_avg_ntwid_sq - this_avg_ntwid**2

        this_avg_data = np.dot(bias_weights, data)
        this_avg_data_sq = np.dot(bias_weights, data_sq)
        this_var_data = this_avg_data_sq - this_avg_data**2

        this_cov_data = np.dot(bias_weights, cov)
        this_cov_data = this_cov_data - this_avg_ntwid*this_avg_data

        avg_ntwid[i] = this_avg_ntwid
        var_ntwid[i] = this_var_ntwid
        avg_data[i] = this_avg_data
        var_data[i] = this_var_data
        cov_data[i] = this_cov_data

    return (neglogpdist, neglogpdist_data, avg_ntwid, var_ntwid, avg_data, var_data, cov_data)



# Get covariance matrix over data, reweighted for each of the beta phi vals
#
# logweights: shape: (N_tot, )   
# ntwid:      shape: (N_tot, )
# data:       shape: (N_dat, N_tot)
def reweight_cov_mat(logweights, ntwid, data, beta_phi_vals):
    
    # Covariance matrix of data for each beta_phi value
    cov_mat = np.zeros((beta_phi_vals.size, data.shape[0], data.shape[0]))

    for i, beta_phi_val in enumerate(beta_phi_vals):
        print("beta phi: {:.2f}".format(beta_phi_val))
        sys.stdout.flush()
        bias_logweights = logweights - beta_phi_val*ntwid
        bias_logweights -= bias_logweights.max()
        norm = np.log(np.sum(np.exp(bias_logweights)))
        bias_logweights -= norm

        bias_weights = np.exp(bias_logweights)

        avg_data = np.dot(data, bias_weights)
        centered_data = data - avg_data[:, None]

        this_cov = np.multiply(centered_data, bias_weights)
        this_cov = np.dot(this_cov, centered_data.T)
        cov_mat[i] = this_cov

    return cov_mat
