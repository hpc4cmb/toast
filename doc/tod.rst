.. _tod:

Telescope TOD
=================================

The TOD base class represents the timestream information describing a telescope that is acquiring data.  The base class enforces a minimal set of methods for reading and writing detector data and flags, detector pointing, and timestamps.  The base class also provide methods for returning information about the data distribution, including which samples are local to a given process.

.. autoclass:: toast.tod.TOD
    :members:

The base TOD class contains a member which is an instance of a Cache object.  This is similar to a dictionary of arrays, but by default the memory used in these arrays is allocated in C, rather than using the python memory pool.  This allows us to do aligned memory allocation and explicitly manage the lifetime of the memory.

.. autoclass:: toast.tod.Cache
    :members:

