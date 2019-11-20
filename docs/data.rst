.. _dist:

Data Model
=================

TOAST works with data organized into *observations*.  Each observation is independent of any other observation.  An observation consists of co-sampled detectors for some span of time.  The intrinsic detector noise is assumed to be stationary within an observation.  Typically there are other quantities which are constant for an observation (e.g. elevation, weather conditions, satellite procession axis, etc).

An observation is just a dictionary with at least one member ("tod") which is an instance of a class that derives from the `toast.TOD` base class.  Every experiment will have their own TOD derived classes, but TOAST includes some built-in ones as well.

The inputs to a TOD class constructor are at least:

1. The detector names for the observation.
2. The number of samples in the observation.
3. The geometric offset of the detectors from the boresight.
4. Information about how detectors and samples are distributed among processes.

.. autoclass:: toast.tod.TOD
    :members:

The TOD class can act as a storage container for different "flavors" of timestreams as well as a source and sink for the observation data (with the read_\*() and write_\*() methods).  The TOD base class has one member which is a `Cache` class.

.. autoclass:: toast.cache.Cache
    :members:

This class looks like a dictionary of numpy arrays, but the memory is allocated outside of Python, which means it can be explicitly managed / freed.  This `cache` member is where alternate flavors of the timestream data are stored.

Each observation can also have a noise model associated with it.  An instance of a Noise class (or derived class) describes the noise properties for all detectors in the observation.

.. autoclass:: toast.tod.Noise
    :members:

The data used by a TOAST workflow consists of a list of observations, and is encapsulated by the `toast.Data` class.

.. autoclass:: toast.dist.Data
    :members:

If you are running with a single process, that process has all observations and all data within each observation locally available.  If you are running with more than one process, the data with be distributed across processes.


Data Distribution
--------------------------

Although you can use TOAST without MPI, the package is designed for data that is
distributed across many processes.  When passing the data through a toast workflow, the data is divided up among processes based on the details of the `toast.Comm` class that is used and also the shape of the process grid in each observation.

A toast.Comm instance takes the global number of processes available (MPI.COMM_WORLD) and divides them into groups. Each process group is assigned one or more observations. Since observations are independent, this means that different groups can be independently working on separate observations in parallel. It also means that inter-process communication needed when working on a single observation can occur with a smaller set of processes.

.. autoclass:: toast.mpi.Comm
    :members:

Just to reiterate, if your `toast.Comm` has multiple process groups, then each group will have an independent list of observations in `toast.Data.obs`.

What about the data *within* an observation?  A single observation is owned by exactly one of the process groups.  The MPI communicator passed to the TOD constructor is the group communicator.  Every process in the group will store some piece of the observation data.  The division of data within an observation is controlled by the `detranks` option to the TOD constructor.  This option defines the dimension of the rectangular "process grid" along the detector (as opposed to time) direction.  Common values of `detranks` are:

    * "1" (processes in the group have all detectors for some slice of time)
    * Size of the group communicator (processes in the group have some of the detectors for the whole time range of the observation)

The detranks parameter must divide evenly into the number of processes in the group communicator.

As a concrete example, imagine that MPI.COMM_WORLD has 24 processes. We split this into 4 groups of 6 procesess. There are 6 observations of varying lengths and every group has one or 2 observations. Here is a picture of what data each process would have. The global process number is shown as well as the rank within the group:

.. image:: _static/toast_data_dist.png

In either case the full dataset is divided into one or more observations, and
each observation has one TOD object (and optionally other objects that describe
the noise, valid data intervals, etc).  The toast "Comm" class has two levels of
MPI communicators that can be used to divide many observations between whole
groups of processes.  In practice this is not always needed, and the default
construction of the Comm object just results in one group with all processes.
