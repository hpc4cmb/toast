.. _opmapmaking:

Map Making Tools
------------------------------

CMB map-making codes and algorithms have a long history going back more than 20 years.
Early work focused on finding the maximum likelihood solution to the noise-weighted
least squares expression for the pixelized map (and other parameters) given the data and
an estimate of the time domain noise.  Later efforts took other approaches, such as the
destriping formalism used in Planck and agressive filtering and binning used in
some ground experiments.  In terms of algorithms built in to TOAST, there are 2 primary tools.  The first is a wrapper around the external libmadam destriping map-maker.  The second is a set of native mapmaking tools which are already quite usable for many common cases.

Libmadam Wrapper
~~~~~~~~~~~~~~~~~~~~

This operator has some basic options and can also take a dictionary of parameters to pass to the underlying Fortran code for complete control over the options.

.. autoclass:: toast.todmap.OpMadam
    :members:


Native Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo:: Document these more fully once the communication performance bottleneck is fixed.

.. autoclass:: toast.todmap.OpMapMaker
    :members:
