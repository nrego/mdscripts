# Adapted from WESTPA code by nrego
# Copyright (C) 2013 Matthew C. Zwier and Lillian T. Chong
#


'''Core classes for creating WESTPA command-line tools'''

from __future__ import print_function, division; __metaclass__ = type

import sys, argparse
import work_managers

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
                    
    def add_args(self, parser):
        '''Add arguments specific to this tool to the given argparse parser.'''
        #westpa.rc.add_args(parser)
        # Add some default args here?
        pass
    
    def process_args(self, args):
        '''Take argparse-processed arguments associated with this tool and deal
        with them appropriately (setting instance variables, etc)'''
        #westpa.rc.process_args(args, config_required = self.config_required)
        pass
                        
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
    
    
        
        