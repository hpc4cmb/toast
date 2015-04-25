"""
Time Ordered Astrophysics Scalable Tools (TOAST) is a software package
designed to allow the processing of data from telescopes that acquire
data as timestreams (rather than images).

The general usage of TOAST consists of two steps:

1.  Create a distributed set of class instances which have methods to
    return detector data and pointing.

2.  Pass this distributed data through one or more "operators", which
    manipulate the data.  Complicated workflows can be scripted by
    chaining different operators together.

3.  Call methods of the various data classes to write any desired outputs.

"""

from mpi4py import MPI

from ._version import __version__

from .dist import Comm, Dist, distribute_uniform, distribute_discrete, distribute_det_samples
from .obs import Obs

__all__ = [
    'Comm', 'Dist', 'Obs', 'distribute_uniform', 'distribute_discrete',
    'distribute_det_samples',
    'Streams', 'StreamsWhiteNoise',
    'Pointing', 
    'Baselines', 
    'Noise',
]

#sys.excepthook = sys_excepthook
#def mpi_excepthook(type, value, traceback):
#    sys_excepthook(type, value, traceback)
#    MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook


