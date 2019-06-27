
# Adapted from WESTPA code
# (Copyright (C) 2013 Matthew C. Zwier and Lillian T. Chong)
#

'''A system for parallel, remote execution of multiple arbitrary tasks.
Much of this, both in concept and execution, was inspired by (and in some 
cases based heavily on) the ``concurrent.futures`` package from Python 3.2,
with some simplifications and adaptations (thanks to Brian Quinlan and his
futures implementation).
'''

import logging
log = logging.getLogger(__name__)

from .core import WorkManager, WMFuture, FutureWatcher


# Import core work managers, which should run most everywhere that
# Python does
from . import serial, processes
from .serial import SerialWorkManager
from .processes import ProcessWorkManager

_available_work_managers = {'serial': SerialWorkManager,
                            'processes': ProcessWorkManager}

# Import MPI work manager if available

try:
    import mpi
    from mpi import MPIWorkManager
except ImportError:
    log.info('MPI work manager not available')
    log.debug('traceback follows', exc_info=True)
else:
    _available_work_managers['mpi'] = MPIWorkManager

from . import environment    
from .environment import make_work_manager
