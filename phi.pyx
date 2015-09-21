import numpy, math


# Caluclates value of coarse-grained gaussian for point at r
#   sigma and cutoff in A. only works around this range - have to see
#   About dynamically adjusting the prefactor
@numpy.vectorize
def phi(double r, double sigma, double cutoff):

    cdef double phic
    cdef double pref

    if (abs(r) >= cutoff):
        return 0.0

    else:

        phic = math.exp(-0.5*(cutoff/sigma)**2)
        pref = 1 / ( (2*math.pi)**(0.5) * sigma * math.erf(cutoff / (2**0.5 * sigma)) - 2*cutoff*phic )

        return pref * ( math.exp(-0.5*(r/sigma)**2) - phic )
