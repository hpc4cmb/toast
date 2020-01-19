.. _opsimnoise:

Simulated Detector Noise
------------------------------

TOAST includes an operator that simulates basic detector noise (1/f spectrum plus white noise), and also supports generating correlated noise by specifying a mixing matrix to combine individual detector streams with common mode sources.

.. autoclass:: toast.tod.OpSimNoise
    :members:
