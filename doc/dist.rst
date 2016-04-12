.. _dist:

Data Distribution
=================================

The toast package is designed for data that is distributed across many
processes.  When passing the data to toast processing routines, you can either
use pre-defined base classes as a container and copy your data into them, or
you can create your own derived classes that provide a standard interface for



.. autoclass:: toast.Comm
    :members:

.. autoclass:: toast.Obs
    :members:

.. autoclass:: toast.Dist
    :members:


Example
-----------

.. literalinclude:: ../pipelines/toast_example_dist.py


