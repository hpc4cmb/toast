"""
Python Time Ordered Astrophysics Scalable Tools (PyTOAST) is a software 
package designed to allow the processing of data from telescopes that acquire
data as timestreams (rather than images).
"""


from ._version import __version__

from .dist import (Comm, Data, distribute_uniform, distribute_discrete, 
    distribute_samples)

from .operator import Operator


#sys.excepthook = sys_excepthook
#def mpi_excepthook(type, value, traceback):
#    sys_excepthook(type, value, traceback)
#    MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook


