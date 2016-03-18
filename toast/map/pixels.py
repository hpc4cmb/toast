# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import Comm, Data
from ..operator import Operator
from ..tod import TOD


class OpLocalPixels(Operator):
    """
    Operator which computes the set of locally hit pixels.

    Args:
        pmat (string): Name of the pointing matrix to use.
    """

    def __init__(self, pmat=None):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        # madam uses time-based distribution
        self._timedist = True
        self._pmat = pmat
        if self._pmat is None:
            self._pmat = TOD.DEFAULT_FLAVOR


    @property
    def timedist(self):
        return self._timedist


    def exec(self, data):
        # the two-level pytoast communicator
        comm = data.comm
        # the global communicator
        cworld = comm.comm_world
        # the communicator within the group
        cgroup = comm.comm_group
        # the communicator with all processes with
        # the same rank within their group
        crank = comm.comm_rank

        # initialize the local pixel set

        local = set()

        for obs in data.obs:
            tod = obs['tod']
            for det in tod.local_dets:
                pixels, weights = tod.read_pmat(name=self._pmat, detector=det, local_start=tod.local_samples[0], n=tod.local_samples[1])
                local.update(set(pixels))

        ret = np.zeros(len(local), dtype=np.int64)
        ret[:] = sorted(local)
        return ret


class DistPixels(object):
    """
    A distributed map with multiple values per pixel.

    Pixel domain data is distributed across an MPI communicator.  each
    process has a local map containing a non-unique, arbitrary slice of
    the global data.  Multiple processes may have copies of the same
    pixels in their local maps.  However, each pixel is uniquely
    "owned" by a single process.  This ownership is used for all
    operations where only a single contribution from each pixel is
    needed (e.g. serializing the data to disk).

    For other operations (e.g. all-to-all accumulation of data), the
    contributions from pixels on all processes are used.  This
    communication can be done either by reduction to the "owner" process
    and then re-broadcast, or with a brute-force all-reduce across all
    processes.

    Args:
        comm (mpi4py.MPI.Comm): the MPI communicator containing all 
            processes.
        size (int): the total number of pixels.
        nnz (int): the number of values per pixel.
        localpix (array): an array mapping local index to global pixel.
    """
    def __init__(self, comm=MPI.COMM_WORLD, size=0, nnz=1, dtype=np.float64, localpix=None):
        self._comm = comm
        self._size = size
        self._nnz = nnz
        self._dtype = dtype
        self._local = local
        if local is None:
            # our local map has all pixels
            self._nlocal = self._size
        else:
            self._nlocal = len(local)
            if local.max() > self._size:
                raise RuntimeError("local pixels out of range")

        # this is the directly-accessible local map
        self.data = np.zeros((self._nlocal, self._nnz), dtype=self._dtype)


    @property
    def comm(self):
        return self._comm

    @property
    def size(self):
        return self._size

    @property
    def nnz(self):
        return self._nnz

    @property
    def dtype(self):
        return self._dtype

    @property
    def localpix(self):
        return self._local


    def read_healpix_fits(self, path):
        # For performance reasons, we can't use healpy to read this
        # map, since we want to read in a buffered way all maps and
        # Bcast.


        
        return


    def write_healpix_fits(self, path):
        raise RuntimeError('writing to healpix FITS not yet implemented')
        return
