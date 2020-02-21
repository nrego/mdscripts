import os, glob
import pymbar
import math

beta = 1 / (k*300)
kappa = beta*0.386



def extract_nstar(dname):

    splits = dname.split('_')
    num = float(splits[-1])

    return num if splits[1] != 'neg' else -num

def logsumexp(vals, axis=None):
    max_val = vals.max(axis=axis)
    vals -= max_val

    return np.log(math.fsum(np.exp(vals))) + max_val

# Gets the (normalized) 2d free energy for given window
def gen_negloghist(x1, x2, b1, b2, logweights):


    negloghist = np.zeros((b1.size-1, b2.size-1))
    negloghist[:] = np.inf
    hist = np.zeros_like(negloghist)

    assign_x1 = np.digitize(x1, b1) - 1
    assign_x2 = np.digitize(x2, b2) - 1

    for i_x1 in range(b1.size-1):

        this_assign_x1 = assign_x1 == i_x1

        if this_assign_x1.sum() == 0:
            continue

        for i_x2 in range(b2.size-1):

            this_assign_x2 = assign_x2 == i_x2
            this_assign = (this_assign_x1 & this_assign_x2)

            if this_assign.sum() == 0:
                continue

            this_logweights = logweights[this_assign]

            assert negloghist[i_x1, i_x2] == np.inf
            negloghist[i_x1, i_x2] = -logsumexp(this_logweights)


    return negloghist


dnames = sorted(glob.glob('nstar_*'))

nstars = np.array( [extract_nstar(dname) for dname in dnames] )

bb_n1 = np.arange(0, 131)
bc_n1 = 0.5*(bb_n1[:-1] + bb_n1[1:])
bb_n2 = np.arange(0, 201)

nn1, nn2 = np.meshgrid(bb_n1, bb_n2, indexing='ij')

f_ks = np.loadtxt("../k_00/d_not/trial_0/f_k_all.dat")

# Biased values, using centered bin vals
# Shape: (bc_n1.size, n_windows)
u_l = 0.5*kappa*(bc_n1[:,None] - nstars)**2

q = f_ks[None,:] - u_l 
max_vals = q.max(axis=1)
q = q - max_vals[:,None]

all_negloghist = np.zeros((f_ks.size, bb_n1.size-1, bb_n2.size-1))

## Make histograms in N_v1, N_v2 for each umbrella
for i, (this_fk, dname) in enumerate(zip(f_ks, dnames)):

    print('doing: {}'.format(dname))
    ntw1, n1, ntw2, n2 = [arr.squeeze() for arr in np.split(np.load('{}/aux_data_pooled.npy'.format(dname)), 4, axis=1)]
    logweights = np.load('{}/all_data.dat.npz'.format(dname))['logweights']

    ds = np.load('{}/negloghist.npz'.format(dname))
    negloghist = ds['negloghist']
    all_negloghist[i] = negloghist

    ## Uncomment to save negloghist's for first go-around
    #####################################################
    #negloghist = gen_negloghist(ntw1, ntw2, bb_n1, bb_n2, logweights)
    #np.savez_compressed('{}/negloghist'.format(dname), negloghist=negloghist, nn1=nn1, nn2=nn2)

np.savez_compressed('all_negloghists', negloghists=all_negloghist, f_ks=f_ks, q=q, bb_n1=bb_n1, bb_n2=bb_n2)



denom = np.log(np.sum(np.exp(q), axis=1)) + max_vals
tot_loghist = np.log(np.exp(-all_negloghist).sum(axis=0)) - denom[:,None]

norm = np.log(np.exp(tot_loghist).sum())
tot_loghist -= norm

tot_negloghist = -tot_loghist

# Integrate out N2
max_vals = tot_loghist.max(axis=1)
pvn = -(np.log(np.sum(np.exp(tot_loghist - max_vals[:,None]), axis=1)) + max_vals)

np.loadtxt('../k_00/d_not/trial_0')

