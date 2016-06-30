import numpy as np
from matplotlib import pyplot
import argparse
import logging
from mdtools import dr

import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 36})
mpl.rcParams.update({'xtick.labelsize': 24})
mpl.rcParams.update({'ytick.labelsize': 24})
mpl.rcParams.update({'axes.titlesize': 40})
#mpl.rcParams.update({'titlesize': 42})
log = logging.getLogger()
log.addHandler(logging.StreamHandler())
'''Perform requested analysis on a bunch of phiout datasets'''
def phiAnalyze(infiles, show, start, end, outfile, conv, S, myrange, nbins):
    phi_vals = np.zeros((len(infiles), 8), dtype=np.float32)
    prev_phi = 0.0
    prev_n = 0
    n_0 = 0
    n_arr = np.empty((len(infiles),2))

    for i, infile in enumerate(infiles):
        log.debug('loading file: {}'.format(infile))
        ds = dr.loadPhi(infile)
        '''
        with open(infile) as f:
            # Ugh
            for li, line in enumerate(f):
                if (li == 2):
                    phi = float(line.split()[3])
                    break
        '''
        phi = ds.phi
        bphi = phi*conv #Could be in kT or kJ/mol
        if (args.plotDist):
            #fig = pyplot.figure()
            #mu = ds.data[start:].mean()[0]
            rng = ds.getRange(start=start, end=end)
            mu = ds.getMean(start=start, end=end, bphi=bphi)
            #var = ds.data[start:].var()[0]
            var = ds.getVar(start=start, end=end, bphi=bphi)
            txtstr = "$\mu={:.3f}$\n$\sigma^2={:.3f}$\n$F={:.2f}$".format(mu, var, var/mu)
            #print(txtstr)
            hist, bounds = np.histogram(np.array(ds.data[start:end]['$\~N$']), bins=50, normed=1)
            ctrs = np.diff(bounds)/2.0 + bounds[:-1]
            pyplot.bar(ctrs, hist, width=np.diff(ctrs)[0])
            pyplot.annotate(txtstr, xy=(0.2,0.75), xytext=(0.2, 0.75),
                            xycoords='figure fraction', textcoords='figure fraction')
            pyplot.suptitle(r'$\beta \phi ={:.2f}$'.format(bphi), fontsize=42)
            #pyplot.legend()
            if outfile:
                stuff = np.zeros((ctrs.shape[0], 2), dtype=float)
                stuff[:,0] = ctrs
                stuff[:,1] = hist
                np.savetxt(outfile, stuff)

        log.debug("Beta*Phi: {:.3f}".format(bphi))
        n = ds.getMean(start=start, end=end, bphi=bphi)
        #n = ds.data[start:]['$\~N$'].mean()
        log.debug("    N: {:.2f}".format(n))
        lg_n = np.log(n)+bphi
        log.debug("    lg(N) + beta*phi: {:.2f}".format(lg_n))
        if i == 0:
            dndphi_neg = 0
            n_0 = n
            delta_phi = 0
        else:
            try:
                dndphi_neg = (prev_n - n) / (bphi - prev_phi)
            except ZeroDivisionError:
                dndphi_neg = 0
            # Suggested delta phi (adaptive method)
            try:
                delta_phi = (n_0/S)/dndphi_neg
            except ZeroDivisionError:
                delta_phi = 0
        try:
            lg_n_negslope = (dndphi_neg/n) - 1
        except ZeroDivisionError:
            lg_n_negslope = -1

        secondCum = ds.getVar(start=start, end=end, bphi=bphi)

        phi_vals[i] = bphi, n, 0, dndphi_neg, lg_n, lg_n_negslope, delta_phi, secondCum
        prev_phi = bphi
        prev_n = n

    for i in xrange(phi_vals.shape[0]):
        phi_vals[i, 2] = np.trapz(phi_vals[:i+1, 1], phi_vals[:i+1, 0])

    # Fill in deriv of lg_beta graph
    # phi_vals[1:,5] = - np.diff(phi_vals[:, 2]) / np.diff(phi_vals[:, 0])
    log.debug("Phi Vals: {}".format(phi_vals))
    if outfile and not args.plotDist:
        np.savetxt(outfile, phi_vals, fmt='%.2f')
    if show:
        if (args.plotDistAll):
            total_stuff = dr.plotHistAll(start=start, end=end, nbins=nbins)
            if outfile:
                np.savetxt(outfile, total_stuff)
        if (args.plotN):
            title = r"$\langle{\~N}\rangle'_\phi$"
            num = 1
            pyplot.plot(phi_vals[:, 0],phi_vals[:, num], 'o-', markersize=12, linewidth=4)
            #pyplot.fill_between(phi_vals[:,0], phi_vals[:,num], color='none', hatch='\\', edgecolor='b')
        if (args.plotInteg):
            title = r'$\int{\langle{N}\rangle_\phi}d\phi$'
            num = 2
            pyplot.plot(phi_vals[:, 0], phi_vals[:, num], 'o-')
        if (args.plotLogN):
            title = r'$\ln{\langle{N}\rangle_\phi} + \beta\phi$'
            num = 4
            pyplot.plot(phi_vals[:, 0],phi_vals[:, num], 'o-')
        if (args.plotNegSlope):
            title = r'$\langle{\delta N^2}\rangle_\phi / \langle{N}\rangle_\phi - 1$'
            num = 5
            pyplot.plot(phi_vals[:, 0],phi_vals[:, num], 'o-')
        if (args.plotSus):
            title = r"$\langle{\delta \~N^2}\rangle'_\phi$"
            num = 7
            pyplot.plot(phi_vals[:, 0], phi_vals[:, num], 'o-', markersize=12, linewidth=4)
        if (args.plotBoth):
            pyplot.close('all')
            f, axarr = pyplot.subplots(2, sharex=True)
            axarr[0].plot(phi_vals[:, 0],phi_vals[:, 1], 'o-')
            axarr[0].set_ylabel(r'$\langle{N}\rangle_\phi$')
            ticks = axarr[0].get_yticks()
            axarr[0].set_yticks(ticks[::2])
            axarr[1].plot(phi_vals[:, 0], phi_vals[:, 7], 'o-')
            axarr[1].set_ylabel(r'$\langle{\delta N^2}\rangle_\phi$')
            ticks = axarr[0].get_yticks()
            axarr[0].set_yticks(ticks[::2])            
            if conv == 1:
                pyplot.xlabel(r'$\phi$ (kJ/mol)')
            else:
                pyplot.xlabel(r'$\beta\phi$ ($k_B T$)')
            pyplot.show()
        if (args.plotDist or args.plotDistAll):
            pyplot.show()
        else:
            #pyplot.plot(phi_vals[:, 0],phi_vals[:, num])
            #pyplot.title(title + r' vs $\beta\phi$')

            if conv == 1:
                pyplot.xlabel(r'$\phi$ (kJ/mol)')
            else:
                pyplot.xlabel(r'$\beta\phi$ ($k_B T$)')

            pyplot.ylabel(title)

            if myrange:
                pyplot.ylim(myrange)

            pyplot.show()

def parseRange(rangestr):
    spl = rangestr.split(",")
    return tuple([float(i) for i in spl])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Phi Reader. Look at N v. time plots at individual Phi values, or \
                                      construct N v. Phi plot from multiple phi values")

    parser.add_argument('input', metavar='INPUT', type=str, nargs='+',
                        help='Input file names')
    parser.add_argument('-o', '--output', default=None,
                        help='print output of N v. Phi plot')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='first timepoint (in ps)')
    parser.add_argument('-e', '--end', type=int, default=None,
                        help='last timepoint (in ps) - default is last available time point')
    parser.add_argument('--debug', action='store_true',
                        help='print debugging info')
    parser.add_argument('-T', metavar='TEMP', type=float,
                        help='convert Phi values to kT, for TEMP (K)')
    parser.add_argument('--plotN', action='store_true',
                        help='plot N v Phi (or Beta Phi) (default no)')
    parser.add_argument('--plotInteg', action='store_true',
                        help='plot integral N v Phi (or Beta Phi) (default no)')
    parser.add_argument('--plotLogN', action='store_true',
                        help='plot (Log N + Phi) v Phi (default no)')
    parser.add_argument('--plotNegSlope', action='store_true',
                        help='plot Negative slope of (Log N + Phi) v Phi (default no)')
    parser.add_argument('--plotSus', action='store_true',
                        help='plot Negative slope of (<N> vs beta phi), the susceptibility')
    parser.add_argument('--plotDist', action='store_true',
                        help='plot water number distributions for each phi value')
    parser.add_argument('--plotDistAll', action='store_true',
                        help='Plot water histograms over all simulations in INPUT (e.g. to gauge overall sampling)')
    parser.add_argument('--plotBoth', action='store_true',
                        help='Plot N v phi and var(N) v phi')
    parser.add_argument('--blockAvg', action='store_true',
                        help='Perform block averaging over one simulation')
    parser.add_argument('-S', type=int,
                        help='Number of phi points for guessing adaptive spacing')
    parser.add_argument('--range', type=str,
                        help="Specify custom range (as 'minY,maxY') for plotting")
    parser.add_argument('--nbins', type=int, default=50,
                        help='Number of bins for histograms')



    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    show = args.plotN or args.plotLogN or args.plotNegSlope or args.plotInteg or args.plotDist or args.plotDistAll or args.plotSus or args.plotBoth
    #infiles = ['phi_{:05d}/phiout.dat'.format(inarg) for inarg in args.input]
    infiles = args.input

    log.debug("{} input files".format(len(infiles)))
    start = args.start
    end = args.end
    outfile = args.output
    conv = 1
    if args.T:
        conv /= (args.T * 0.008314)

    S = args.S or len(infiles) # Number of phi values - used to suggest delta_phi using adaptive method

    myrange = None
    if args.range:
        myrange = parseRange(args.range)

    if len(infiles) == 1 and args.blockAvg:
        ds = dr.loadPhi(infiles[0], corr_len=1)
        block_vals = ds.blockAvg(start,end=end)
        data_len = np.array(ds.data[start:end]['$\~N$']).shape[0]
        print "Data length:{}".format(data_len)
        pyplot.plot(block_vals[:,0], np.sqrt(block_vals[:,2]), 'ro')
        pyplot.xlim(0,block_vals.shape[0]/2)
        pyplot.xlabel("Block size")
        pyplot.ylabel(r'$\sigma_{{\langle{\~N}\rangle}}$')
        pyplot.show()
        if outfile:
            block = input("Input block size for output: ")
            ds.printOut(outfile, start=start, block=block)
    elif len(infiles) == 1 and not args.plotDist:
        dr.loadPhi(infiles[0])
        dr.plot(start=start, end=end, ylim=myrange)
        dr.show()
    else:
        phiAnalyze(infiles, show, start, end, outfile, conv, S, myrange, args.nbins)
