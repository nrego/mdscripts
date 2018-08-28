from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import argparse
from scipy import interpolate
from scipy.signal import savgol_filter
from IPython import embed

mpl.rcParams.update({'axes.labelsize': 20})
mpl.rcParams.update({'xtick.labelsize': 20})
mpl.rcParams.update({'ytick.labelsize': 20})
mpl.rcParams.update({'axes.titlesize': 25})
mpl.rcParams.update({'legend.fontsize':18})


def get_diffs(arr):
    fprime = np.zeros_like(arr)
    fprime[...] = np.nan
    fcurve = np.zeros_like(arr)
    fcurve[...] = np.nan

    fprime[1:] = np.diff(arr) 
    fcurve[2:] = np.diff(fprime[1:])

    return fprime, fcurve

def run(args):

    dat = np.loadtxt(args.input)
    bb = bb = dat[:,0]
    fvn = dat[:,1]
    
    smooth_fvn = savgol_filter(fvn, args.window, 3)
    smooth_fprime = savgol_filter(fvn, args.window, 3, deriv=1)
    smooth_fcurve = savgol_filter(fvn, args.window, 3, deriv=2)
    fprime, fcurve = get_diffs(fvn)

    outarr = np.dstack((bb, fvn, smooth_fvn, fprime, smooth_fprime, fcurve, smooth_fcurve)).squeeze()
    outhead = 'N    F(N)    [smooth]F(N)    F\'(N)    [smooth] F\'(N)    F\'\'(N)    [smooth] F\'\'(N) '
    # Output data
    np.savetxt('{}gradf.dat'.format(args.outprefix), outarr, header=outhead)


    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6,10))

    ax1.plot(bb, fvn, label='raw data')
    ax1.plot(bb, smooth_fvn, '--', label='smoothed')

    ax2.plot(bb, fprime, label='finite difference')
    ax2.plot(bb, smooth_fprime, '--', label='smoothed')

    ax3.plot(bb, fcurve, label='finite difference')
    ax3.plot(bb, smooth_fcurve, '--', label='smoothed')

    ax3.set_xlabel(r'$N$')
    ax1.set_ylabel(r'$\beta F_V(N)$')
    ax2.set_ylabel(r'$\frac{\partial \beta F_V(N)}{\partial N}$')
    ax3.set_ylabel(r'$\frac{\partial^2 \beta F_V(N)}{\partial N^2}$')

    fig.tight_layout()
    ax1.legend()
    ax2.legend()
    ax3.legend()

    fig.savefig('{}fgrad.pdf'.format(args.outprefix))

    if args.interactive:
        plt.show()



if __name__=='__main__':

    description= '''\
                Calculate and output (smoothed) finite differences of a free energy profile 
                using Savitsky-Gavloy filtering algorithm

                Input file should be formatted as shape (n_bins, 2)
                where first column is the bin values (i.e. N) and the second is F(N)


                Will output (PDF) image of F(N), Fprime(N), and Fprimeprime(N), along with smoothed curves, to
                   '[prefix]fgrad.pdf' and '[prefix]fgrad.dat', where 'prefix' is an optional prepended-prefix.

                The datafile (...dat) is formatted as shape: (n_bins, 4); columns are:
                    N  F(N)   (smoothed) Fprime(N)    (smoothed) Fprimeprime(N)
                '''
    parser = argparse.ArgumentParser( description=description)

    parser.add_argument('-f', '--input', metavar='INFILE', type=str, default='neglogpdist.dat',
                        help='Input data file name')
    parser.add_argument('-w', '--window', type=int, default=31, 
                        help='Length of smoothing window (default: 31: i.e. 15 in either direction)')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='If true, display plots interactively (default: False)')
    parser.add_argument('--outprefix', type=str, 
                        help='Prepend prefix string to output figure and datasets')
    parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)

