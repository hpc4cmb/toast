.. _maptools:

Map-making Tools
=================================

This broad class of operations include anything that generates pixel-space data products.


Distributed Pixel-space Data
--------------------------------

.. autoclass:: toast.map.OpLocalPixels
    :members:

.. autoclass:: toast.map.DistPixels
    :members:


Diagonal Noise Covariance
------------------------------

.. autoclass:: toast.map.OpInvCovariance
    :members:

.. autofunction:: toast.map.covariance_invert

.. autofunction:: toast.map.covariance_rcond


Native Mapmaking
-----------------------

Still to-do.  See `git milestone here <https://github.com/tskisner/pytoast/milestones/Native%20Mapmaking%20Tools>`_


External Madam Interface
----------------------------------

.. autoclass:: toast.map.OpMadam
    :members:

