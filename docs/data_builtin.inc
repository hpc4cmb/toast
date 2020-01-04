.. _databuiltin:

Built-in Data Support
-----------------------------

In most cases of "real" experiments / telescopes, there will be custom TOAST classes to represent the data.  However, TOAST comes with support for some (mainly simulated) data types.  These are useful when simulating proposed experiments that do not yet have detailed specifications.  Once a project reaches the level of having detailed hardware specifications (either designed or measured), then it should implement custom classes for that instrument description and data I/O or simulation.

Simulated TOD Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall that a ``TOD`` class represents the properties of some detectors from a telescope for one observation.  This includes its physical location / motion through space as well as geometric locations of the detectors and (in the case of real data) methods to read detector data.  For simulated TOD classes, we simulate the telescope boresight pointing and detector properties.  We do **not** simulate detector data as part of the TOD class- that is done by various simulation operators that accumulate signal and noise from various sources.  For generic satellite simulations, we have:

.. autofunction:: toast.todmap.slew_precession_axis

.. autofunction:: toast.todmap.satellite_scanning

.. autoclass:: toast.todmap.TODSatellite
    :members:

For generic ground-based simulations we have:

.. autoclass:: toast.todmap.TODGround
    :members:


Simulated Noise Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One common noise model that is frequency used in simulations is a 1/f spectrum with white noise.

.. autoclass:: toast.tod.AnalyticNoise
    :members:


Simulated Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulating regular data intervals is common task for many simulations:

.. autofunction:: toast.tod.regular_intervals
