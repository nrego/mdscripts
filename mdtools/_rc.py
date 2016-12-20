# Adapted from westpa rc

from __future__ import division, print_function; __metaclass__ = type

import logging
log = logging.getLogger('mdtools.rc')

import os, sys, errno
import mdtools
from work_managers import SerialWorkManager


class MDTOOLSRC:
    '''A class, an instance of which is accessible as ``mdtools.rc``, to handle global issues for mdtools code,
    such as loading modules and plugins, writing output based on verbosity level, adding default command line options,
    and so on.'''
    
        
    def __init__(self):        
        self.verbosity = None
        self.process_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        
        self.work_manager = SerialWorkManager()
        
        self.status_stream = sys.stdout
        
    def add_args(self, parser):
        group = parser.add_argument_group('general options')
        
        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument('--quiet', dest='verbosity', action='store_const', const='quiet',
                             help='emit only essential information')
        egroup.add_argument('--verbose', '-v', dest='verbosity', action='store_const', const='verbose',
                             help='emit extra information')
        egroup.add_argument('--debug', dest='verbosity', action='store_const', const='debug',
                            help='enable extra checks and emit copious information')
        
    @property
    def verbose_mode(self):
        return (self.verbosity in ('verbose', 'debug'))
    
    @property
    def debug_mode(self):
        return (self.verbosity == 'debug')
    
    @property
    def quiet_mode(self):
        return (self.verbosity == 'quiet')
                            
    def process_args(self, args, config_required = True):
        self.cmdline_args = args
        self.verbosity = args.verbosity
        

        self.config_logging()
                    
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
                          'loggers': {'mdtools': {'handlers': ['console'], 'propagate': False},
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
        
    def pstatus(self, *args, **kwargs):
        fileobj = kwargs.pop('file', self.status_stream)
        if kwargs.get('termonly', False) and not fileobj.isatty():
            return
        if self.verbosity != 'quiet':
            print(*args, file=fileobj, **kwargs)
            
    def pstatus_term(self, *args, **kwargs):
        fileobj = kwargs.pop('file', self.status_stream)
        if fileobj.isatty() and self.verbosity != 'quiet':
            print(*args, file=fileobj, **kwargs)
        
    def pflush(self):
        for stream in (self.status_stream, sys.stdout, sys.stderr):
            try:
                stream.flush()
            except AttributeError:
                pass
