.. _dist:

Data Distribution
=================================

The toast package is designed for data that is distributed across many
processes.  When passing the data to toast processing routines, you can either
use pre-defined base classes as a container and copy your data into them, or
you can create your own derived classes that provide a standard interface.

In either case the full dataset is divided into one or more observations, and
each observation has one TOD object (and optionally other objects that describe
the noise, valid data intervals, etc).  The toast "Comm" class has two levels of
MPI communicators that can be used to divide many observations between whole
groups of processes.  In practice this is not always needed, and the default
construction of the Comm object just results in one group with all processes.

.. autoclass:: toast.mpi.Comm
    :members:

The Data class below is essentially just a list of observations for each
process group.

.. autoclass:: toast.dist.Data
    :members:


Example
-----------

.. literalinclude:: ../examples/toast_example_dist.py
