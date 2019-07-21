## Basic two layer NN for sam surface (input vec is all positions)

from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import hexagdly
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from IPython import embed

import argparse