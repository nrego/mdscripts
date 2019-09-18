## My set of WHAM/MBAR utilities
from __future__ import division

import numpy as np



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
#   Returns:   weights  (n_tot, ) [weight for each datapoint]
def gen_data_logweights(bias_mat, f_k, n_samples):

    Q = f_k - bias_mat + np.log(n_samples)
    max_vals = Q.max(axis=1)
    Q -= max_vals[:,None]

    logweights = -( np.log(np.exp(Q).sum(axis=1) ) + max_vals )
    logweights -= logweights.max()

    return logweights

# This will only work for linear interpolated du/dl for now...
def gen_pdist_xvg(dudl, bias_mat, n_samples, f_k, lmbdas, binbounds):

    n_lambdas = lmbdas.size

    pdist = np.zeros(binbounds.shape[0]-1, dtype=np.float64)
    for n_idx in range(dudl.size):
        denom_arr = f_k - bias_mat[n_idx, :]
        
        denom_arr = n_samples * np.exp(denom_arr)
        denom = denom_arr.sum()

        val = dudl[n_idx]
        bin_assign = (val >= binbounds[:-1]) * (val < binbounds[1:])

        pdist[bin_assign] += 1.0/denom

    return pdist

# U[i,j] is exp(-beta * Uj(n_i))
# This is equivalent to the bias mat in whamerr.py, right??
def gen_U_nm(all_dat, nsims, beta, start, end=None):

    n_tot = all_dat.shape[0]

    u_nm = np.zeros((n_tot, nsims))

    for i, (ds_name, ds) in enumerate(dr.datasets.items()):
        u_nm[:, i] = np.exp( -beta*(0.5*ds.kappa*(all_dat-ds.Nstar)**2 + ds.phi*all_dat) )

    return np.matrix(u_nm)

# Log likelihood
def kappa(xweights, bias_mat, nsample_diag, ones_m, ones_N, n_tot):
    
    f = np.append(0, xweights)
    
    # n_tot x n_w
    Q = f - bias_mat

    # n_w x 1
    diag = np.array(np.log(np.dot(nsample_diag, ones_m))).squeeze()
    P = Q + diag
    c = P.max(axis=1)
    P -= c[:,None]

    ln_sum_exp = np.log(np.exp(P).sum(axis=1)) + c

    logLikelihood = (ones_N.T/n_tot)*ln_sum_exp[:,None] - \
                    ones_m.T * nsample_diag * f[:,None]

    return logLikelihood.item()

def grad_kappa(xweights, bias_mat, nsample_diag, ones_m, ones_N, n_tot):

    #u_nm = np.exp(-bias_mat)
    f = np.append(0, xweights)

    # n_tot x n_w
    Q = f - bias_mat

    # n_w x 1
    diag = np.array(np.log(np.dot(nsample_diag, ones_m))).squeeze()
    P = Q + diag
    c = P.max(axis=1)
    P -= c[:,None]

    ln_sum_exp = np.log(np.exp(P).sum(axis=1)) + c

    denom = np.exp(ln_sum_exp[:,None])

    W = np.exp(Q)/denom

    grad = nsample_diag*W.T*(ones_N/n_tot) - nsample_diag*ones_m

    return ( np.array(grad[1:]) ).squeeze()

def hess_kappa(xweights, bias_mat, nsample_diag, ones_m, ones_N, n_tot):
    
    raise NotImplementedError

def callbackF(xweights):
    global Nfeval
    #log.info('Iteration {}'.format(Nfeval))
    log.info('\rIteration {}\r'.format(Nfeval))
    sys.stdout.flush()
    #log.info('.')

    Nfeval += 1