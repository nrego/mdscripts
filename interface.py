#Instantaneous interface

import numpy
import argparse
import logging

import MDAnalysis

# Caluclates value of coarse-grained gaussian for point at r
#   sigma and cutoff in nm
def phi(r, sigma=0.24, cutoff=0.7):

