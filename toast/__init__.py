
from mpi4py import MPI


'''
TOAST
==============

Time Ordered Astrophysics Scalable Tools (TOAST) is a software package
designed to allow the processing of...
'''

from _version import __version__

#sys.excepthook = sys_excepthook
#def mpi_excepthook(type, value, traceback):
#    sys_excepthook(type, value, traceback)
#    MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook


