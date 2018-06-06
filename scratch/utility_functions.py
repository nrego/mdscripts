import numpy as np


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
