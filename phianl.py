import numpy
from matplotlib import pyplot
import argparse
import logging
from datareader import dr

import matplotlib as mpl

mpl.rcParams.update({'axes.labelsize': 30})
mpl.rcParams.update({'xtick.labelsize': 18})
mpl.rcParams.update({'ytick.labelsize': 18})
mpl.rcParams.update({'axes.titlesize': 36})
#mpl.rcParams.update({'titlesize': 42})


def blockAvg(ds, start, end=None):
    data = numpy.array(ds.data[start+1:end]['$\~N$'])
    #data = ds
    data_var = data.var()
    n_obs = len(data)  # Total number of observations

    #blocks = (numpy.power(2, xrange(int(numpy.log2(n_obs))))).astype(int)
    blocks = numpy.arange(1,len(data)/2+1,1)

    n_blocks = len(blocks)

    block_vals = numpy.zeros((n_blocks, 3))
    block_vals[:, 0] = blocks.copy()

    block_ctr = 0

    for block in blocks:
        n_block = int(n_obs/block)
        obs_prop = numpy.zeros(n_block)

        for i in xrange(n_block):
            ibeg = i*block
            iend = ibeg + block
            obs_prop[i] = data[ibeg:iend].mean()

        block_vals[block_ctr, 1] = obs_prop.mean()
        block_vals[block_ctr, 2] = obs_prop.var() / (n_block-1)
        #block_vals[block_ctr, 2] = (numpy.power(obs_prop,2).mean() - obs_prop.mean()**2).sum() / (n_block - 1)

        block_ctr += 1

    return block_vals

'''Perform requested analysis on a bunch of phiout datasets'''
def phiAnalyze(infiles, show, start, end, outfile, conv, S, myrange):
    phi_vals = numpy.zeros((len(infiles), 8), dtype=numpy.float32)
    prev_phi = 0.0
    prev_n = 0
    n_0 = 0
    n_arr = numpy.empty((len(infiles),2))

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
            ds.data[start:end].hist(bins=20, normed=1)
            pyplot.annotate(txtstr, xy=(0.2,0.75), xytext=(0.2, 0.75),
                            xycoords='figure fraction', textcoords='figure fraction')
            pyplot.suptitle(r'$\beta \phi ={}$'.format(bphi), fontsize=42)
            #pyplot.legend()

        log.debug("Beta*Phi: {:.3f}".format(bphi))
        n = ds.getMean(start=start, end=end, bphi=bphi)
        #n = ds.data[start:]['$\~N$'].mean()
        log.debug("    N: {:.2f}".format(n))
        lg_n = numpy.log(n)+bphi
        log.debug("    lg(N) + beta*phi: {:.2f}".format(lg_n))
        if i == 0:
            dndphi_neg = 0
            n_0 = n
            delta_phi = 0
        else:
            dndphi_neg = (prev_n - n) / (bphi - prev_phi)
            # Suggested delta phi (adaptive method)
            delta_phi = (n_0/S)/dndphi_neg

        lg_n_negslope = (dndphi_neg/n) - 1

        secondCum = ds.getVar(start=start, end=end, bphi=bphi)

        phi_vals[i] = bphi, n, 0, dndphi_neg, lg_n, lg_n_negslope, delta_phi, secondCum
        prev_phi = bphi
        prev_n = n

    for i in xrange(phi_vals.shape[0]):
        phi_vals[i, 2] = numpy.trapz(phi_vals[:i+1, 1], phi_vals[:i+1, 0])

    # Fill in deriv of lg_beta graph
    # phi_vals[1:,5] = - numpy.diff(phi_vals[:, 2]) / numpy.diff(phi_vals[:, 0])
    log.debug("Phi Vals: {}".format(phi_vals))
    if outfile:
        numpy.savetxt(outfile, phi_vals, fmt='%.2f')
    if show:
        if (args.plotDistAll):
            dr.plotHistAll(start=start, end=end, nbins=20)
            dr.show()
        if (args.plotN):
            title = r'$\langle{N}\rangle_\phi$'
            num = 1
            pyplot.plot(phi_vals[:, 0],phi_vals[:, num], 'bo-')
        if (args.plotInteg):
            title = r'$\int{\langle{N}\rangle_\phi}d\phi$'
            num = 2
            pyplot.plot(phi_vals[:, 0], phi_vals[:, num])
        if (args.plotLogN):
            title = r'$\ln{\langle{N}\rangle_\phi} + \beta\phi$'
            num = 4
            pyplot.plot(phi_vals[:, 0],phi_vals[:, num])
        if (args.plotNegSlope):
            title = r'$\langle{\delta N^2}\rangle_\phi / \langle{N}\rangle_\phi - 1$'
            num = 5
            pyplot.plot(phi_vals[:, 0],phi_vals[:, num])
        if (args.plotSus):
            title = r'$\langle{\delta N^2}\rangle_\phi$'
            num = 7
            pyplot.plot(phi_vals[:, 0], phi_vals[:, num])
        if (args.plotDist):
            pyplot.show()
        else:
            #pyplot.plot(phi_vals[:, 0],phi_vals[:, num])
            pyplot.title(title + r' vs $\beta\phi$')

            if conv == 0.001:
                pyplot.xlabel(r'$\phi$ (kJ/mol)')
            else:
                pyplot.xlabel(r'$\beta\phi$')

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
    parser.add_argument('--blockAvg', action='store_true',
                        help='Perform block averaging over one simulation')
    parser.add_argument('-S', type=int,
                        help='Number of phi points for guessing adaptive spacing')
    parser.add_argument('--range', type=str,
                        help="Specify custom range (as 'minY,maxY') for plotting")

    log = logging.getLogger('phianl')


    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    show = args.plotN or args.plotLogN or args.plotNegSlope or args.plotInteg or args.plotDist or args.plotDistAll or args.plotSus
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
        ds = dr.loadPhi(infiles[0], corr_len=10)
        block_vals = blockAvg(ds, start=start)
        pyplot.plot(9000/block_vals[:,0], numpy.sqrt(block_vals[:,2]), 'ro')
        pyplot.xlim(0,500)
        pyplot.xlabel("Block size (ps)")
        pyplot.ylabel(r'$\sigma_{{\langle{\~N}\rangle}}$')
        pyplot.show()
    elif len(infiles) == 1 and not args.plotDist:
        dr.loadPhi(infiles[0])
        dr.plot(start=start, end=end, ylim=myrange)
        dr.show()
    else:
        phiAnalyze(infiles, show, start, end, outfile, conv, S, myrange)
