
from mpi4py import MPI


'''
TOAST
==============

Time Ordered Astrophysics Scalable Tools (TOAST) is a software package
designed to allow the processing of...
'''

from ._version import __version__

from .dist import Comm, Dist
from .obs import Obs

__all__ = [
    'Comm', 'Dist', 'Obs', 'Streams', 'Pointing', 'Baselines', 'Noise',
]

#sys.excepthook = sys_excepthook
#def mpi_excepthook(type, value, traceback):
#    sys_excepthook(type, value, traceback)
#    MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook


