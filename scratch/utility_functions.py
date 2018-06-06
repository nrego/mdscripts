import numpy as np
from constants import k

'''
Calculate the neglogpdist [F(N)] for this bootstrap sample
'''
def get_neglogpdist(all_data, all_data_N, boot_indices, boot_logweights):

    weights = np.exp(boot_logweights)
    weights /= weights.sum()

    max_N = np.ceil(all_data.max())+1
    binbounds = np.arange(0, max_N, 1)

    boot_data = all_data[boot_indices]

    hist, bb = np.histogram(boot_data, bins=binbounds, weights=weights)

    neglogpdist = -np.log(hist)
    neglogpdist -= neglogpdist.min()


    return neglogpdist, bb


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

    hist, bb = np.histogram(boot_data_N, bins=binbounds, weights=weights)

    neglogpdist = -np.log(hist)
    neglogpdist -= neglogpdist.min()

    # Find <N>, <\delta N^2> v phi
    phi_vals = np.arange(0, 10.1, 0.1)
    avg_ns = np.zeros_like(phi_vals)
    var_ns = np.zeros_like(phi_vals)

    boot_data_sq = boot_data**2
    beta = 1/(k*300)

    for i, phi in enumerate(phi_vals):
        bias = beta*phi*boot_data

        this_logweights = boot_logweights - bias
        this_logweights -= this_logweights.max()
        this_weights = np.exp(this_logweights)
        this_weights /= this_weights.sum()

        this_avg_n = np.dot(this_weights, boot_data)
        this_avg_n_sq = np.dot(this_weights, boot_data_sq)
        this_var_n = this_avg_n_sq - this_avg_n**2

        avg_ns[i] = this_avg_n
        var_ns[i] = this_var_n

    phi_dat_arr = np.dstack((phi_vals, avg_ns, var_ns)).squeeze()

    return (neglogpdist, bb, phi_dat_arr)
