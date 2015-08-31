import numpy
from matplotlib import pyplot
import argparse
import logging
from datareader import DataReader

def plotHist(dr, myrange):
    i = 1
    for title, ds in dr.datasets.items():

        if ds.partner is not None:
            ds2 = dr.datasets[ds.partner]
            pyplot.figure(i)
            ds.plot()
            ds2.plot()
            pyplot.legend()
            i += 1

    dr.show()


def parseRange(rangestr):
    spl = rangestr.split(",")
    return tuple([float(i) for i in spl])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="histogram.xvg reader (output from g_bar with option -oh). Examine and \
                                      plot histograms for different lambda's during insertion/deletion")

    parser.add_argument('input', metavar='INPUT', type=str, default='histogram.xvg',
                        help='histogram.xvg input file')
    parser.add_argument('--debug', action='store_true',
                        help='print debugging info')
    parser.add_argument('--range', type=str,
                        help="Specify custom range (as 'minY,maxY') for plotting")

    log = logging.getLogger()
    log.addHandler(logging.StreamHandler())

    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    infile = args.input

    myrange = None
    if args.range:
        myrange = parseRange(args.range)

    dr = DataReader
    dr.loadXVG(infile)

    plotHist(dr, myrange)
