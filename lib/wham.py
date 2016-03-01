## My set of WHAM/MBAR utilities

import np as np

# U[i,j] is exp(-beta * Uj(n_i))
def genU_nm(all_data, nsims, beta, start, end=None):

    n_tot = all_data.shape[0]

    u_nm = np.zeros((n_tot, nsims))

    for i, (ds_name, ds) in enumerate(dr.datasets.iteritems()):
        u_nm[:, i] = np.exp( -beta*(0.5*ds.kappa*(all_data-ds.Nstar)**2 + ds.phi*all_data) )

    return np.matrix(u_nm)

# Log likelihood
def kappa(xweights, u_nm, nsample_diag, ones_m, ones_N, n_tot):

    logf = np.append(0, xweights)
    f = np.exp(-logf) # Partition functions relative to first window

    Q = np.dot(u_nm, np.diag(f))

    logLikelihood = (ones_N.transpose()/n_tot)*np.log(Q*nsample_diag*ones_m) + \
                    np.dot(np.diag(nsample_diag), logf)


    return float(logLikelihood)

def gradKappa(xweights, u_nm, nsample_diag, ones_m, ones_N, n_tot):

    logf = np.append(0, xweights)
    f = np.exp(-logf) # Partition functions relative to first window

    Q = np.dot(u_nm, np.diag(f))
    denom = (Q*nsample_diag).sum(axis=1)

    W = Q/denom

    grad = -nsample_diag*W.transpose()*(ones_N/n_tot) + nsample_diag*ones_m

    return ( np.array(grad[1:]) ).reshape(len(grad)-1 )

def callbackF(xweights):
    global Nfeval
    #log.info('Iteration {}'.format(Nfeval))
    log.info('\rIteration {}\r'.format(Nfeval))
    sys.stdout.flush()
    #log.info('.')

    Nfeval += 1