import numpy as np

from scratch.neural_net.lib import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import argparse

import os


## Base class for command line drivers ##
## This class handles argument parsing, as well as base cmd-line arguments we'll always use.
##   (number of epochs, early-stopping, etc)
## 
## Since this class is meant to be inhereted, its argument building and argument parsing
#    routines are structured to climb down the inheritance tree when adding/parsing cmd
#    line args (so you can add additional args ad naseum by implementing 'add_args' for subclasses).
#
#  All subclasses will call the method "main()" which will in turn call:
#    ->make_parser_and_process(), which will call:
#         [create parser]
#        ->parser = make_parser(), which will create an argparser and add arguments to it by
#            Running 'add_args()' thru the inheritance chain (via _add_all_args(parser))
#         [extract all cmd line args]
#        ->args = parser.parse_args()
#         [interpret arguments and initialize]
#        ->parser_args(args), which will interpret args for each subclass by running down
#            down the inheritance chain via _process_all_args(args)
class Core:

    prog = None
    usage = None
    description = None
    epilog = None

    def __init__(self):
        self.parser = None
        self.args = None

        self.process_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    def _add_all_args(self, parser):
        '''Add arguments for all components from which this class derives to the given parser,
        starting with the class highest up the inheritance chain (most distant ancestor).'''
        self.parser = parser
        for cls in reversed(self.__class__.__mro__):
            try:
                fn = cls.__dict__['add_args']
            except KeyError:
                pass
            else:
                fn(self,parser)
    
    def _process_all_args(self, args):
        self.args = args
        '''Process arguments for all components from which this class derives,
        starting with the class highest up the inheritance chain (most distant ancestor).'''
        for cls in reversed(self.__class__.__mro__):
            try:
                fn = cls.__dict__['process_args']
            except KeyError:
                pass
            else:
                fn(self,args)


    def add_args(self, parser):
        '''Add some default arguments'''
        group = parser.add_argument_group('General training options')
        group.add_argument("--infile", "-f", type=str, default="sam_pattern_pooled.npz",
                           help="Input file name containing SAM pattern dataset (Default: %(default)s)")
        group.add_argument("--augment-data", action="store_true",
                           help="Augment data by flipping every input pattern over all 6-fold symetries (Default: Do not augment data)")
        group.add_argument("--batch-size", type=int, default=200,
                           help="Size of training batches. There will be (N_data//batch_size + N_data mod batch_size) batches in each "\
                                "training epoch.  (Default: %(default)s)")
        group.add_argument("--n-valid", type=int, default=5,
                           help="Number of partitions for cross-validation (Default: split data into %(default)s groups")
        group.add_argument("--skip-cv", action="store_true",
                           help="If true, do not perform cross-validation (Default: %(default)s).")
        group.add_argument("--learning-rate", type=float, default=0.001,
                           help="Learning rate for training (Default: %(default)s)")
        group.add_argument("--n-patience", type=int, default=None,
                           help="Maximum number of epochs to tolerate where CV performance decreases before breaking out of training."\
                                "Default: No break-out.")
        group.add_argument("--n-epochs", type=int, default=2000,
                           help="Number of training epochs (Default: %(default)s)")
        group.add_argument("--break-out", type=float, default=None,
                           help="Break out of training if CV MSE falls below this value."\
                                "Default: No break-out.")

    def process_args(self, args):
        self.cmdlineargs = args # Keep a record of all input arguments for posterity

        self.infile = args.infile
        self.augment_data = args.augment_data
        self.cv_nets = []

        self.n_epochs = args.n_epochs
        self.break_out = args.break_out
        self.skip_cv = args.skip_cv

        self.n_patience = args.n_patience
        self.batch_size = args.batch_size
        self.n_valid = args.n_valid
        self.learning_rate = args.learning_rate

    def make_parser(self, prog=None, usage=None, description=None, epilog=None):
        prog = prog or self.prog
        usage = usage or self.usage
        description = description or self.description
        epilog = epilog or self.epilog
        parser = argparse.ArgumentParser(prog=prog, usage=usage, description=description, epilog=epilog,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         conflict_handler='resolve')
        self._add_all_args(parser)
        return parser

    def make_parser_and_process(self):
        '''convenience method to make parser, fill it with args, parse args, and send them to children to process'''
        
        # Initializes argparser, and adds all arguments by calling add_all_args
        parser = self.make_parser()
        # Get command line args
        args = parser.parse_args()
        # Distribute them to subclasses and process/initialize
        self._process_all_args(args)

        return args

    def run(self):
        ''' Run the main routines of tool - each subclass must overwrite this method'''
        raise NotImplementedError

    def main(self):
        '''Every subclass (tool) should call this method from within a __name__='__main__' block'''

        self.make_parser_and_process()
        self.run()


## All command-line tools for training ANNs should subclass this!
class NNDriver(Core):
    ''' Basic class for command line tools that train NN's.  This guy encapsulates NN architecture, including whether we want a CNN'''

    def __init__(self):
        super(NNDriver, self).__init__()

        self.net_train = []
        self.trainers = []
        self.net_final = []


    def add_args(self, parser):
        group = parser.add_argument_group("NN Architecture options")
        group.add_argument("--n-hidden-layer", type=int, default=3,
                           help="Number of hidden layers (network depth). (Default: %(default)s)")
        group.add_argument("--n-node-hidden", type=int, default=18,
                           help="Number of nodes per hidden layer (network width). (Default: %(default)s)")
        group.add_argument("--n-node-feature", type=int, default=0,
                           help="Number of nodes in an optional 'feature' layer between the hidden layers and the output layer."\
                                 "This can be used to train the network to 'learn' features that are fed in to a linear model (via the output node)."\
                                 "If '0', this layer is ignored and the hidden layers feed directly into the output layer (default: %(default)s)")
        group.add_argument("--drop-out", type=float, default=0.0,
                           help="Dropout probability per node during training. Can assist with model robustness. (Default %(default)s)")
        group.add_argument("--do-conv", action="store_true",
                           help="Do a convolutional neural net (default: false)")
        group.add_argument("--n-conv-filters", type=int, default=4,
                           help="Number of convolutional filters to apply; ignored if not doing CNN (Default: %(default)s)")
        group.add_argument("--no-run", action="store_true",
                           help="Don't run CV if true")

    def process_args(self, args):
        self.n_hidden_layer = args.n_hidden_layer
        self.n_node_hidden = args.n_node_hidden if self.n_hidden_layer > 0 else 0
        self.n_node_feature = args.n_node_feature
        self.drop_out = args.drop_out

        self.do_conv = args.do_conv
        self.n_conv_filters = args.n_conv_filters

        self.no_run = args.no_run

        if self.n_hidden_layer == 0 and not self.do_conv:
            raise ValueError("If zero layers, must do CNN")


