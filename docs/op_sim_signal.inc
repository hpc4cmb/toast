.. _opsimsignal:

Simulated Detector Signal
------------------------------

Every experiment is different, and a realistic simulation will likely require some
customized code to model the instrument.  However some kinds of simulations are generic,
especially when doing trade studies in the early design phase of a project.  TOAST comes
with several built-in operators for simulating detector signals.

Sky Model
~~~~~~~~~~~~~~

"Sky" In this case are sources of signals coming from outside of the Earth.  The most simplistic model of the sky is just an input map at the same pixelization that is used for analysis (to avoid pixelization effects).  There can be one map per detector or one map for a set of detectors (for example if simulating all detectors in a band without including bandpass variability):

.. autoclass:: toast.todmap.OpSimScan
    :members:

The cosmological and orbital dipole can be simulated with this operator:

.. autoclass:: toast.todmap.OpSimDipole
    :members:

TOAST can also use some external packages for more complicated sky simulations.  One of these is PySM, which supports generating bandpass-integrated sky maps in memory for each detector from a set of component maps.  It can also do basic smoothing of the input signal:

.. autoclass:: toast.todmap.OpSimPySM
    :members:

For studies of far side lobes and full 4Pi beams, TOAST can use the external libconviqt package to convolve beam a_lm's with a sky a_lm at every sample:

.. autoclass:: toast.todmap.OpSimConviqt
    :members:


Atmosphere Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although many ground experiments have traditionally modelled atmosphere as "correlated
noise", in TOAST we take a different approach.  For each observation, a realization of
the physical atmosphere slab is created and moved according to the wind speed.  Every
detector sample is formed by a line of sight integral through this slab.  This sort of
real-space modelling naturally leads to correlated signal in the time domain that
depends on atmospheric parameters and focalplane geometry.

.. autoclass:: toast.todmap.OpSimAtmosphere
    :members:

Other Signals
~~~~~~~~~~~~~~~~~

Any systematic contaminant that is constant (for an observation) in the scanning reference frame (as opposed to sky coordinates), can be simulated as "Scan Synchronous Signal":

.. autoclass:: toast.todmap.OpSimScanSynchronousSignal
    :members:

This next operator can be used to apply random gain errors to the timestreams:

.. autoclass:: toast.tod.OpGainScrambler
    :members:
