.. _sim:

Simulations
=================================

There are several classes included in pytoast that can simulate different types of data.

Simulated Telescope
-----------------------

.. autoclass:: toast.tod.TODHpixSpiral
    :members:


.. autofunction:: toast.tod.slew_precession_axis

.. autofunction:: toast.tod.satellite_scanning

.. autoclass:: toast.tod.TODSatellite
    :members:


Simulated Noise Model
---------------------------

.. autoclass:: toast.tod.AnalyticNoise
    :members:


Simulated Intervals
---------------------------

.. autofunction:: toast.tod.regular_intervals


Simulated Detector Data
--------------------------

.. autoclass:: toast.tod.OpSimNoise
    :members:

.. autoclass:: toast.tod.OpSimGradient
    :members:

.. autoclass:: toast.tod.OpSimScan
    :members:

This operator uses an externally installed libconviqt.

.. autoclass:: toast.tod.OpSimConviqt
    :members:


