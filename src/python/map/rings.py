# Copyright (c) 2017-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import healpy as hp
import numpy as np

try:
    import libsharp
    available = True
except ModuleNotFoundError:
    libsharp = None
    available = False

import timemory
from ..cache import Cache
from ..mpi import MPI


def expand_pix(startpix, ringpix, local_npix, local_pix):
    """Turn first pixel index and number of pixel in full array of pixels

    to be optimized with cython
    """
    autotimer = timemory.auto_timer()
    i = 0
    for start, num in zip(startpix, ringpix):
        local_pix[i:i + num] = np.arange(start, start + num)
        i += num


def distribute_rings(nside, rank, n_mpi_processes):
    """Create a libsharp map distribution based on rings

    Build a libsharp grid object to distribute a HEALPix map
    balancing North and South distribution of rings to achieve
    the best performance on Harmonic Transforms
    Returns the grid object and the pixel indices array in RING ordering

    Parameters
    ---------

    nside : int
        HEALPix NSIDE parameter of the distributed map
    rank, n_mpi_processes, ints
        rank of the current MPI process and total number of processes

    Returns
    -------

    grid : libsharp.healpix_grid
        libsharp object that includes metadata about HEALPix distributed rings
    local_pix : np.ndarray
        integer array of local pixel indices in the current MPI process in RING
        ordering
    """
    if libsharp is None:
        raise RuntimeError('libsharp not available')
    autotimer = timemory.auto_timer()
    nrings = 4 * nside - 1  # four missing pixels

    # ring indices are 1-based
    ring_indices_emisphere = np.arange(2 * nside, dtype=np.int32) + 1

    local_ring_indices = ring_indices_emisphere[rank::n_mpi_processes]

    # to improve performance, symmetric rings north/south need to be in the same rank
    # therefore we use symmetry to create the full ring indexing

    if local_ring_indices[-1] == 2 * nside:
        # has equator ring
        local_ring_indices = np.concatenate(
            [local_ring_indices[:-1],
             nrings - local_ring_indices[::-1] + 1]
        )
    else:
        # does not have equator ring
        local_ring_indices = np.concatenate(
            [local_ring_indices,
             nrings - local_ring_indices[::-1] + 1]
        )

    grid = libsharp.healpix_grid(nside, rings=local_ring_indices)
    return grid, local_ring_indices


class DistRings(object):
    """
    A map with unique pixels distributed as disjoint isolatitude rings.

    Designed for Harmonic Transforms with libsharp

    Pixel domain data is distributed across an MPI communicator.  Each
    process has a number of isolatitude rings and a list of the pixels
    within those rings.

    Args:
        comm (mpi4py.MPI.Comm): the MPI communicator containing all
            processes.
        size (int): the total number of pixels.
        nnz (int): the number of values per pixel.
        submap (int): the locally stored data is in units of this size.
        local (array): the list of local submaps (integers).
        localpix (array): the list of local pixels (integers).
        nest (bool): nested pixel order flag
    """

    def __init__(self, comm=MPI.COMM_WORLD, nnz=1, dtype=np.float64,
                 nside=16):
        if libsharp is None:
            raise RuntimeError('libsharp not available')
        self.data = None
        self._comm = comm
        self._nnz = nnz
        self._dtype = dtype
        self._nest = False
        self._nside = nside

        self._cache = Cache()

        self._libsharp_grid, self._local_ring_indices = distribute_rings(
            self._nside, self._comm.rank, self._comm.size)
        # returns start index of the ring and number of pixels
        startpix, ringpix, _, _, _ = hp.ringinfo(
            self._nside, self._local_ring_indices.astype(np.int64))

        local_npix = self._libsharp_grid.local_size()
        self._local_pixels = self._cache.create(
            "local_pixels", shape=(local_npix,), type=np.int64)
        expand_pix(startpix, ringpix, local_npix, self._local_pixels)

        self.data = self._cache.create(
            "data", shape=(local_npix, self._nnz), type=self._dtype)

    def __del__(self):
        if self.data is not None:
            del self.data
        del self._local_pixels
        self._cache.clear()

    @property
    def comm(self):
        """
        (mpi4py.MPI.Comm): The MPI communicator used.
        """
        return self._comm

    @property
    def nnz(self):
        """
        (int): The number of non-zero values per pixel.
        """
        return self._nnz

    @property
    def dtype(self):
        """
        (numpy.dtype): The data type of the values.
        """
        return self._dtype

    @property
    def nested(self):
        """
        (bool): If True, data is HEALPix NESTED ordering.
        """
        return self._nest

    @property
    def local_pixels(self):
        """
        (numpy.ndarray int64): Array of local pixel indices in RING ordering
        """
        return self._local_pixels

    @property
    def libsharp_grid(self):
        """
        (libsharp grid): Libsharp grid distribution object
        """
        return self._libsharp_grid
