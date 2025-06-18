# Copyright (c) 2024-2025 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import astropy.io.fits as af
import healpy as hp
import numpy as np
from astropy.wcs import WCS

from .pixels import PixelData, PixelDistribution
from .pixels_io_healpix import read_healpix


def footprint_distribution(
    healpix_nside=None,
    healpix_nside_submap=None,
    healpix_submap_file=None,
    healpix_coverage_file=None,
    wcs_coverage_file=None,
    comm=None,
):
    """Create a PixelDistribution from a pre-defined sky footprint.

    Usually a PixelDistribution object is created by passing through the detector
    pointing and determining the locally hit submaps.  However, this can be expensive
    if the data must be loaded from disk and if there is insufficient memory to hold
    the detector data in a persistent way.

    This function provides a way for building a PixelDistribution where all processes
    have the full footprint locally, regardless of whether their local detector
    pointing hits all submaps.  For high resolution sky products with many processes
    per node, use of shared memory may be required.

    Only certain combinations of options are supported:

    1.  If `wcs_coverage_file` is specified, that is taken to be the WCS projection
        of the coverage.  The number of pixels is set by the extent of the WCS, NOT
        the actual pixel values.  The number of submaps is set to one.  All healpix
        options should be None.
    2.  If `healpix_coverage_file` is specified, the NSIDE of the file is used to
        define the number of pixels and the non-zero pixel values along with
        `healpix_nside_submap` is used to compute the nonzero submaps in this coverage.
        The same hit submaps are used across all processes.
    3.  If `healpix_submap_file` is specified, non-zero values represent the hit
        submaps.  `healpix_nside` is then used to define the NSIDE and the number of
        pixels.
    4.  If neither file is specified, `healpix_nside` is used to define the NSIDE and
        number of pixels.  `healpix_nside_submap` is used to compute the number of
        submaps.  All submaps are considered hit in this case.

    Args:
        healpix_nside (int):  If specified, the NSIDE of the coverage map.
        healpix_nside_submap (int):  If specified, the NSIDE of the submaps.
        healpix_coverage_file (str):  The path to a coverage map.
        healpix_submap_file (str):  The path to a map with the submaps to use.
        wcs_coverage_file (str):  The path to a WCS coverage map in the primary HDU.
        comm (MPI.Comm):  The MPI communicator or None.

    Returns:
        (PixelDistribution): The output pixel distribution.

    """
    rank = 0
    if comm is not None:
        rank = comm.rank

    wcs = None
    nest = True
    if wcs_coverage_file is not None:
        # Load a WCS projection
        if (
            healpix_nside is not None
            or healpix_nside_submap is not None
            or healpix_coverage_file is not None
            or healpix_submap_file is not None
        ):
            msg = "If loading a wcs coverage file, all other options should be None"
            raise RuntimeError(msg)
        n_pix = None
        if rank == 0:
            hdulist = af.open(wcs_coverage_file)
            n_pix = np.prod(hdulist[0].data.shape)
            wcs = WCS(hdulist[0].header)
            hdulist.close()
            del hdulist
        if comm is not None:
            n_pix = comm.bcast(n_pix, root=0)
            wcs = comm.bcast(wcs, root=0)
        n_submap = 1
        local_submaps = [0]
    elif healpix_coverage_file is not None:
        if healpix_nside_submap is None:
            msg = "You must specify the submap NSIDE to use with the coverage file"
            raise RuntimeError(msg)
        n_pix = None
        n_submap = None
        local_submaps = None
        if rank == 0:
            hpix_data = read_healpix(healpix_coverage_file, field=(0,), nest=nest)
            nside = hp.get_nside(hpix_data)
            n_pix = 12 * nside**2
            n_submap = 12 * healpix_nside_submap**2

            # Find hit pixels
            hit_pixels = np.logical_and(
                hpix_data != 0,
                hp.mask_good(hpix_data),
            )
            unhit_pixels = np.logical_not(hit_pixels)

            # Set map data to one or zero so we can find hit submaps
            hpix_data[hit_pixels] = 1
            hpix_data[unhit_pixels] = 0

            # Degrade to submap resolution
            submap_data = hp.ud_grade(
                hpix_data, healpix_nside_submap, order_in="NEST", order_out="NEST"
            )

            # Find hit submaps
            hit_submaps = submap_data > 0
            local_submaps = np.arange(12 * healpix_nside_submap**2, dtype=np.int32)[
                hit_submaps
            ]
        if comm is not None:
            n_pix = comm.bcast(n_pix, root=0)
            n_submap = comm.bcast(n_submap, root=0)
            local_submaps = comm.bcast(local_submaps, root=0)
    elif healpix_submap_file is not None:
        if healpix_nside is None:
            msg = "You must specify the coverage NSIDE to use with the submap file"
            raise RuntimeError(msg)
        n_pix = None
        n_submap = None
        local_submaps = None
        if rank == 0:
            submap_data = read_healpix(healpix_submap_file, field=(0,), nest=nest)
            nside_submap = hp.npix2nside(len(submap_data))
            n_submap = 12 * nside_submap**2
            n_pix = 12 * healpix_nside**2

            # Find hit submaps
            hit_submaps = np.logical_and(
                submap_data != 0,
                hp.mask_good(submap_data),
            )
            local_submaps = np.arange(n_submap, dtype=np.int32)[hit_submaps]
        if comm is not None:
            n_pix = comm.bcast(n_pix, root=0)
            n_submap = comm.bcast(n_submap, root=0)
            local_submaps = comm.bcast(local_submaps, root=0)
    else:
        if healpix_nside is None:
            msg = "No files specified, you must set healpix_nside"
            raise RuntimeError(msg)
        if healpix_nside_submap is None:
            msg = "No files specified, you must set healpix_nside_submap"
            raise RuntimeError(msg)
        n_pix = 12 * healpix_nside**2
        n_submap = 12 * healpix_nside_submap**2
        local_submaps = np.arange(n_submap, dtype=np.int32)
    dist = PixelDistribution(
        n_pix=n_pix, n_submap=n_submap, local_submaps=local_submaps, comm=comm
    )
    if wcs is None:
        dist.nest = nest
    else:
        dist.wcs = wcs
    return dist
