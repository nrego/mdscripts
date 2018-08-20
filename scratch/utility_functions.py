import numpy as np
from constants import k
from whamutils import get_neglogpdist
from IPython import embed
'''
Calculate the neglogpdist [F(N)] for this bootstrap sample
'''
def fn_neglogpdist(all_data, all_data_N, boot_indices, boot_logweights):

    max_N = np.ceil( max(all_data.max(), all_data_N.max()) ) + 1
    binbounds = np.arange(0, max_N, 1)

    boot_data = all_data[boot_indices]
    boot_data_N = all_data_N[boot_indices]

    neglogpdist = nget_neglogpdist(boot_data, binbounds, boot_logweights)

    neglogpdist_N = nget_neglogpdist(boot_data_N, binbounds, boot_logweights)


    return neglogpdist, neglogpdist_N, binbounds


'''
get avg n and var n with phi, as well as neglogpdist
'''
def get_n_v_phi(all_data, all_data_N, boot_indices, boot_logweights):

    weights = np.exp(boot_logweights)
    weights /= weights.sum()

    # Find -ln P(N)
    max_N = np.ceil(all_data_N.max()) + 1
    binbounds = np.arange(0, max_N, 1)

    boot_data = all_data[boot_indices]
    boot_data_N = all_data_N[boot_indices]

    #hist, bb = np.histogram(boot_data_N, bins=binbounds, weights=weights)
    neglogpdist = get_neglogpdist(boot_data_N.astype(np.float64), binbounds, boot_logweights)

    # Find <N>, <\delta N^2> v phi
    phi_vals = np.arange(0, 10.1, 0.1)
    avg_ns = np.zeros_like(phi_vals)
    var_ns = np.zeros_like(phi_vals)

    boot_data_N_sq = boot_data_N**2
    beta = 1/(k*300)

    for i, phi in enumerate(phi_vals):
        bias = beta*phi*boot_data

        this_logweights = boot_logweights - bias
        this_logweights -= this_logweights.max()
        this_weights = np.exp(this_logweights)
        this_weights /= this_weights.sum()

        this_avg_n = np.dot(this_weights, boot_data_N)
        this_avg_n_sq = np.dot(this_weights, boot_data_N_sq)
        this_var_n = this_avg_n_sq - this_avg_n**2

        avg_ns[i] = this_avg_n
        var_ns[i] = this_var_n

    phi_dat_arr = np.dstack((phi_vals, avg_ns, var_ns)).squeeze()

    return (neglogpdist, bb, phi_dat_arr)

def get_2d_rama(all_data, all_data_N, boot_indices, boot_logweights):

    weights = np.exp(boot_logweights)
    weights /= weights.sum()

    binbounds = np.arange(-180,187,4)

    phi_boot = all_data[boot_indices, 0]
    psi_boot = all_data[boot_indices, 1]

    hist, bb1, bb2 = np.histogram2d(phi_boot, psi_boot, binbounds, weights=weights)

    neglogpdist = -np.log(hist)
    neglogpdist -= neglogpdist.min()

    return (neglogpdist, binbounds)
    
def get_weighted_data(all_data, all_data_N, boot_indices, boot_logweights):

    boot_data = all_data[boot_indices]
    boot_data_N = all_data_N[boot_indices]

    return (boot_logweights, boot_data, boot_data_N)


