.. _intervals:

Data Intervals
=================================

Within each TOD object, a process contains some local set of detectors and range of samples.  That range of samples may contain one or more contiguous "chunks" that were used when distributing the data.  Separate from this data distribution, TOAST has the concept of valid data "intervals".  This list of intervals applies to the whole TOD sample range, and all processes have a copy of this list.

.. autoclass:: toast.tod.Interval
    :members:

