## My set of WHAM/MBAR utilities

import numpy as np

# Generate a probability distribution over a variable by integrating
# This currently works for phi datasets **ONLY**
def gen_pdist(all_data, bias_mat, n_samples, logweights, data_range, nbins):
    bias_mat = np.exp(-bias_mat)
    bias_mat = np.array(bias_mat)
    range_min, range_max = data_range
    nstep = float(range_max - range_min) / nbins
    #binbounds = np.arange(range_min, range_max+nstep, nstep)
    binbounds = np.linspace(range_min, range_max, nbins+1)

    weights_prod_nsample = n_samples * np.exp(logweights)

    pdist = np.zeros(nbins, dtype=np.float64)

    for n_idx in xrange(all_data.shape[0]):
        if all_data.ndim == 2:
            for k in xrange(all_data.shape[1]):
                val = all_data[n_idx, k]
                bin_assign = (val >= binbounds[:-1]) * (val < binbounds[1:])
                denom = np.dot(weights_prod_nsample, bias_mat[n_idx, :])
                pdist[bin_assign] += 1.0/denom

        elif all_data.ndim == 1:
            val = all_data[n_idx]

            # which bin does ntwid fall into? A boolean array.
            # Might want to assert sum==1 here; i.e. it falls into exactly 1 bin
            bin_assign = (val >= binbounds[:-1]) * (val < binbounds[1:])

            denom = np.dot(weights_prod_nsample, bias_mat[n_idx, :])

            pdist[bin_assign] += 1.0/denom

    return binbounds, pdist

def gen_pdist_xvg(all_data, bias_mat, n_samples, logweights, data_range, nbins):
    pass

# U[i,j] is exp(-beta * Uj(n_i))
# This is equivalent to the bias mat in whamerr.py, right??
def gen_U_nm(all_data, nsims, beta, start, end=None):

    n_tot = all_data.shape[0]

    u_nm = np.zeros((n_tot, nsims))

    for i, (ds_name, ds) in enumerate(dr.datasets.iteritems()):
        u_nm[:, i] = np.exp( -beta*(0.5*ds.kappa*(all_data-ds.Nstar)**2 + ds.phi*all_data) )

    return np.matrix(u_nm)

# Log likelihood
def kappa(xweights, bias_mat, nsample_diag, ones_m, ones_N, n_tot):
    #u_nm = np.exp(-bias_mat)
    f = np.append(0, xweights)
    #z = np.exp(-f)

    #Q = np.dot(u_nm, np.diag(z))

    Q = bias_mat + f
    #Q = np.exp(-Q)
    #logLikelihood = (ones_N.transpose()/n_tot)*np.log(Q*nsample_diag.diagonal().T) + \
    #                np.dot(np.diag(nsample_diag), f)
    logLikelihood = (ones_N.transpose()/n_tot)*np.log(( np.exp(-Q + np.log(nsample_diag.diagonal())) ).sum(axis=1)) + \
                    np.dot(np.diag(nsample_diag), f)


    return float(logLikelihood)

def grad_kappa(xweights, bias_mat, nsample_diag, ones_m, ones_N, n_tot):

    #u_nm = np.exp(-bias_mat)
    f = np.append(0, xweights)
    #z = np.exp(-f) # Partition functions relative to first window

    #Q = np.dot(u_nm, np.diag(z))

    Q = bias_mat + f
    #Q = np.exp(-Q)
    #denom = (Q*nsample_diag).sum(axis=1)
    denom = ( np.exp(-Q + np.log(nsample_diag.diagonal())) ).sum(axis=1) 

    W = np.exp(-Q)/denom

    grad = -nsample_diag*W.transpose()*(ones_N/n_tot) + nsample_diag*ones_m

    return ( np.array(grad[1:]) ).reshape(len(grad)-1 )

def callbackF(xweights):
    global Nfeval
    #log.info('Iteration {}'.format(Nfeval))
    log.info('\rIteration {}\r'.format(Nfeval))
    sys.stdout.flush()
    #log.info('.')

    Nfeval += 1