from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import argparse
from scipy.interpolate import interp1d
from IPython import embed

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 25})
mpl.rcParams.update({'legend.fontsize':18})

def window_smoothing(arr, windowsize=5):
    ret_arr = np.zeros_like(arr)
    ret_arr[:] = float('nan')

    for i in range(ret_arr.size):
        ret_arr[i] = arr[i-windowsize:i+windowsize].mean()

    return ret_arr

def run(args):


    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6,10))

    dat = np.loadtxt(args.input)
    n_vals = bb = dat[:,0]
    neglogpdist = dat[:,1]

    smoothed_f = window_smoothing(neglogpdist, windowsize=args.window)
    #embed()
    # Derivative of F(N) w.r.t. N
    fprime = np.diff(smoothed_f)
    # ... and smooth this derivative
    smoothed_fprime = window_smoothing(fprime, windowsize=args.window)

    # Second derivative
    fcurve = np.diff(smoothed_fprime)
    # .... and smooth it
    smoothed_fcurve = window_smoothing(fcurve, windowsize=args.window)

    ax1.plot(n_vals, neglogpdist)

    ax2.plot(n_vals[:-1], fprime)
    ax2.plot(n_vals[:-1], smoothed_fprime)

    ax3.plot(n_vals[:-2], fcurve)
    ax3.plot(n_vals[:-2], smoothed_fcurve)

    ax3.set_xlabel(r'$N$')
    ax1.set_ylabel(r'$\beta F_V(N)$')
    ax2.set_ylabel(r'$\frac{\partial \beta F_V(N)}{\partial N}$')
    ax3.set_ylabel(r'$\frac{\partial^2 \beta F_V(N)}{\partial N^2}$')

    fig.tight_layout()

    fig.savefig('{}fgrad.pdf'.format(args.outprefix))

    if args.interactive:
        plt.show()

    # Find the minimum curvarture
    min_idx = np.nanargmin(smoothed_fcurve)
    min_val = smoothed_fcurve[min_idx]
    min_n = n_vals[min_idx]

    print("min: Nstar={:0.4f},  curvarture_star={:0.4f}".format(min_n, min_val))

    headerstr = 'N   F(N)   [smoothed] Fprime(N)   [smoothed] Fprimeprime(N)  [smoothing window: {}]'.format(args.window)

    outarr = np.empty((n_vals.size, 4))
    outarr[:,0] = n_vals
    outarr[:,1] = neglogpdist
    outarr[1:,2] = smoothed_fprime
    outarr[2:,3] = smoothed_fcurve

    np.savetxt('{}fgrad.dat'.format(args.outprefix), outarr, header=headerstr)
    np.savetxt('{}min_val.dat'.format(args.outprefix), [min_n, min_val])


if __name__=='__main__':

    description= '''\
                Calculate and output (smoothed) finite differences of a free energy profile 

                Input file should be formatted as shape (n_bins, 2)
                where first column is the bin values (i.e. N) and the second is F(N)


                Will output (PDF) image of F(N), Fprime(N), and Fprimeprime(N), along with smoothed curves, to
                   '[prefix]fgrad.pdf' and '[prefix]fgrad.dat', where 'prefix' is an optional prepended-prefix.

                The datafile (...dat) is formatted as shape: (n_bins, 4); columns are:
                    N  F(N)   (smoothed) Fprime(N)    (smoothed) Fprimeprime(N)
                '''
    parser = argparse.ArgumentParser( description=description)

    parser.add_argument('-f', '--input', metavar='INFILE', type=str, default='neglogpdist_N.dat',
                        help='Input data file name')
    parser.add_argument('-w', '--window', type=int, default=5, 
                        help='Length of smoothing window (default: 5 in either direction)')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='If true, display plots interactively (default: False)')
    parser.add_argument('--outprefix', type=str, 
                        help='Prepend prefix string to output figure and datasets')
    parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)

