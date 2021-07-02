.. _opprocessing:

Timestream Processing
------------------------------

Many timestream manipulations done prior to map-making are very specific to the instrument.  However there are a few operations that are generically useful.

Filtering
~~~~~~~~~~~~~~~~~~~~

This operator is used to build a template in azimuth bins of signal that is fixed in the scanning reference frame.  This template is then subtracted from the timestream.

.. autoclass:: toast.todmap.OpGroundFilter
    :members:

This next operator fits a polynomial to each scan and subtracts it.

.. autoclass:: toast.tod.OpPolyFilter
    :members:

Calibration
~~~~~~~~~~~~~~~~~

This operator applies a set of gains to the timestreams:

.. autoclass:: toast.tod.OpApplyGain
    :members:

Crosstalk
~~~~~~~~~

.. automodule:: toast.tod.crosstalk
    :members: add_crosstalk_args, SimpleCrosstalkMatrix, OpCrosstalk

Utilities
~~~~~~~~~~~~~~~~~~~~~

These operators are used to manipulate cached data or perform other helper functions.

.. autoclass:: toast.tod.OpFlagGaps
    :members:

.. autoclass:: toast.tod.OpFlagsApply
    :members:

.. autoclass:: toast.tod.OpMemoryCounter
    :members:
