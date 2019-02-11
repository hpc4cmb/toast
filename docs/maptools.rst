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

.. autoclass:: toast.map.OpAccumDiag
    :members:

.. autofunction:: toast.map.covariance_invert

.. autofunction:: toast.map.covariance_rcond

.. autofunction:: toast.map.covariance_multiply

.. autofunction:: toast.map.covariance_apply


Native Mapmaking
-----------------------

Using the distributed diagonal noise covariance tools, one can make a simple
binned map.  Porting the old TOAST map-maker to this version of TOAST is still
on the to-do list.


External Madam Interface
----------------------------------

If the MADAM library is installed and in your shared library search path, then
you can use it to make maps.

.. autoclass:: toast.map.OpMadam
    :members:

