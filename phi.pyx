import numpy, math


# Caluclates value of coarse-grained gaussian for point at r
#   sigma and cutoff in A. only works around this range - have to see
#   About dynamically adjusting the prefactor
@numpy.vectorize
def phi(double r_sq, double sigma, double sigma_sq, double cutoff, double cutoff_sq):

    cdef double phic
    cdef double pref

    if (abs(r_sq) >= cutoff_sq):
        return 0.0

    else:
        sigma_sq = sigma*sigma
        cutoff_sq = cutoff*cutoff
        phic = math.exp(-0.5*(cutoff_sq/sigma_sq))
        pref = 1 / ( (2*math.pi)**(0.5) * sigma * math.erf(cutoff / (2**0.5 * sigma)) - 2*cutoff*phic )

        return (pref * ( math.exp((-0.5*r_sq)/(sigma_sq)) - phic ))**3
