# Adapted from WESTPA code by nrego
# Copyright (C) 2013 Matthew C. Zwier and Lillian T. Chong
#


'''Core classes for creating WESTPA command-line tools'''

from __future__ import print_function, division; __metaclass__ = type

import sys, argparse
import work_managers

import mdtools

import os

import logging
log = logging.getLogger(__name__)

class ToolComponent:
    '''Base class for command line tools and components used in constructing tools'''
    
    def __init__(self):
        self.config_required = False
        self.include_args = {}
        self.arg_defaults = {}
        self.parser = None
        self.args = None
        
    def include_arg(self, argname):
        self.include_args[argname] = True
        
    def exclude_arg(self, argname):
        self.include_args[argname] = False
        
    def set_arg_default(self, argname, value):
        self.arg_defaults[argname] = value
        
    def add_args(self, parser):
        '''Add arguments specific to this component to the given argparse parser.'''
        pass
    
    def process_args(self, args):
        '''Take argparse-processed arguments associated with this component and deal
        with them appropriately (setting instance variables, etc)'''
        pass

    def add_all_args(self, parser):
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
    
    def process_all_args(self, args):
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

class Tool(ToolComponent):
    '''Base class for command line tools'''
    
    prog = None
    usage = None
    description = None
    epilog = None
    
    def __init__(self):
        super(Tool,self).__init__()
        self.process_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
                    
    def add_args(self, parser):
        '''Add arguments specific to this tool to the given argparse parser.'''
        mdtools.rc.add_args(parser)
        # Add some default args here?
                            
    def process_args(self, args, config_required = True):
        mdtools.rc.process_args(args)
                        
    def make_parser(self, prog=None, usage=None, description=None, epilog=None, args=None):
        prog = prog or self.prog
        usage = usage or self.usage
        description = description or self.description
        epilog = epilog or self.epilog
        parser = argparse.ArgumentParser(prog=prog, usage=usage, description=description, epilog=epilog,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         conflict_handler='resolve')
        self.add_all_args(parser)
        return parser
            
    def make_parser_and_process(self, prog=None, usage=None, description=None, epilog=None, args=None):
        '''A convenience function to create a parser, call add_all_args(), and then call process_all_args().
        The argument namespace is returned.'''
        parser = self.make_parser(prog,usage,description,epilog,args)
        args = parser.parse_args(args)
        self.process_all_args(args)
        return args
    
    def go(self):
        '''Perform the analysis associated with this tool.'''
        raise NotImplementedError
    
    def main(self):
        '''A convenience function to make a parser, parse and process arguments, then call self.go()'''
        self.make_parser_and_process()
        self.go()

    def config_logging(self):
        import logging.config
        logging_config = {'version': 1, 'incremental': False,
                          'formatters': {'standard': {'format': '-- %(levelname)-8s [%(name)s] -- %(message)s'},
                                         'debug':    {'format': '''\
-- %(levelname)-8s %(asctime)24s PID %(process)-12d TID %(thread)-20d
   from logger "%(name)s" 
   at location %(pathname)s:%(lineno)d [%(funcName)s()] 
   ::
   %(message)s
'''}},
                          'handlers': {'console': {'class': 'logging.StreamHandler',
                                                   'stream': 'ext://sys.stdout',
                                                   'formatter': 'standard'}},
                          'loggers': {'west': {'handlers': ['console'], 'propagate': False},
                                      'westpa': {'handlers': ['console'], 'propagate': False},
                                      'oldtools': {'handlers': ['console'], 'propagate': False},
                                      'westtools': {'handlers': ['console'], 'propagate': False},
                                      'westext': {'handlers': ['console'], 'propagate': False},
                                      'work_managers': {'handlers': ['console'], 'propagate': False},
                                      'py.warnings': {'handlers': ['console'], 'propagate': False}},
                          'root': {'handlers': ['console']}}
        
        logging_config['loggers'][self.process_name] = {'handlers': ['console'], 'propagate': False}
            
        if self.verbosity == 'debug':
            logging_config['root']['level'] = 5 #'DEBUG'
            logging_config['handlers']['console']['formatter'] = 'debug'
        elif self.verbosity == 'verbose':
            logging_config['root']['level'] = 'INFO'
        else:
            logging_config['root']['level'] = 'WARNING'

        logging.config.dictConfig(logging_config)
        logging_config['incremental'] = True
        logging.captureWarnings(True)
    
                
class ParallelTool(Tool):
    '''Base class for command-line tools parallelized with wwmgr. This automatically adds and processes
    wwmgr command-line arguments and creates a work manager at self.work_manager.'''

    def __init__(self, wm_env=None):
        super(ParallelTool,self).__init__()
        self.work_manager = None
        self.wm_env = wm_env or work_managers.environment.default_env
        self.max_queue_len = None

    def make_parser_and_process(self, prog=None, usage=None, description=None, epilog=None, args=None):
        '''A convenience function to create a parser, call add_all_args(), and then call process_all_args().
        The argument namespace is returned.'''
        parser = self.make_parser(prog,usage,description,epilog,args)
        self.wm_env.add_wm_args(parser)
        
        args = parser.parse_args(args)
        self.wm_env.process_wm_args(args)
        
        # Instantiate work manager        
        self.work_manager = self.wm_env.make_work_manager()
        
        # Process args
        self.process_all_args(args)
        return args

    def add_args(self, parser):
        pgroup = parser.add_argument_group('parallelization options')
        pgroup.add_argument('--max-queue-length', type=int,
                            help='''Maximum number of tasks that can be queued. Useful to limit RAM use
                            for tasks that have very large requests/response. Default: no limit.''')
    
    def process_args(self, args):
        self.max_queue_len = args.max_queue_length
        log.debug('max queue length: {!r}'.format(self.max_queue_len))

    def go(self):
        '''Perform the analysis associated with this tool.'''
        raise NotImplementedError
    
    def main(self):
        '''A convenience function to make a parser, parse and process arguments, then run self.go() in the master process.'''
        self.make_parser_and_process()
        with self.work_manager:
            if self.work_manager.is_master:
                self.go()
            else:
                self.work_manager.run()
    

class Subcommand(ToolComponent):
    '''This kind of mimics Tool (see below), except that it creates a new subparser and adds its args to that 
    It hooks into a 'subparsers' instance (returned by the master parser's parser.add_subparsers command)
    '''

    subcommand = None
    help_text = None
    description = None

    def __init__(self, parent):
        # not running the constructor on ToolComponent here, I guess...?
        self.parent = parent
        self.subparser = None

    def add_subparser(self, subparsers):
        subparser = subparsers.add_parser(self.subcommand, help=self.help_text, description=self.description)

        self.add_all_args(subparser)
        # Anthing processing args will be able to this instance as args.subcommand
        subparser.set_defaults(subcommand=self)
        self.subparser = subparser

    @property
    def work_manager(self):
        '''get work manager from subcommand's parent. Raises AttributeError if parent does not exist of if parent does not support parallelization
        (i.e. parent subclasses Tool, not ParallelTool)'''
        return self.parent.work_manager

    @property
    def max_queue_len(self):
        '''Raises an AttributeError if parent is not parallel'''
        return self.parent.max_queue_len

    @property
    def n_workers(self):
        try:
            return self.work_manager.n_workers
        except AttributeError:
            return 1
    
    
