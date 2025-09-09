# Map Domain

Internally, map domain objects (hit maps, signal maps, pixel noise covariances, etc) are
stored in a distributed fashion.  This enables each process to only store a subset of
the sky pixels, which saves memory for large maps at high resolution.  `PixelData`
objects have methods to read and write HDF5 and FITS format files.

::: toast.pixels.PixelDistribution

::: toast.pixels.PixelData

Map domain objects can be saved to and loaded from different formats on disk depending
on the pixelization.  Healpix maps support both FITS and a more performant HDF5 format.
Maps in WCS flat projections only support FITS formats.

::: toast.pixels_io_healpix.read_healpix
::: toast.pixels_io_healpix.write_healpix

::: toast.pixels_io_wcs.write_wcs
::: toast.pixels_io_wcs.read_wcs
