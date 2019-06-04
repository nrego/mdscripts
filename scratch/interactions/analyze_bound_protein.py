from __future__ import division, print_function
import numpy as np
from scipy.spatial import cKDTree
import MDAnalysis
import argparse

from mdtools import MDSystem

parser = argparse.ArgumentParser('Analyze bound state of protein-protein complex')
