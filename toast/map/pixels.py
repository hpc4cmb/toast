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
                pixels, weights = tod.read_pmat(name=self._pmat, detector=det, local_start=0, n=tod.local_samples[1])
                local.update(set(pixels))

        ret = np.zeros(len(local), dtype=np.int64)
        ret[:] = sorted(local)
        return ret


class DistPixels(object):
    """
    A distributed map with multiple values per pixel.

    Pixel domain data is distributed across an MPI communicator.  each
    process has a local data stored in one or more "submaps".  The size
    of the submap can be tuned to balance storage (smaller submap size
    means fewer wasted pixels stored) and ease of indexing (larger
    submap means faster global-to-local pixel lookups).

    Although multiple processes may have the same submap of data stored
    locally, the lowest-rank process that has a given submap is the
    "owner" for operations like serialization. 

    Args:
        comm (mpi4py.MPI.Comm): the MPI communicator containing all 
            processes.
        size (int): the total number of pixels.
        nnz (int): the number of values per pixel.
        submap (int): the locally stored data is in units of this size.
        local (array): the list of local submaps (integers).
    """
    def __init__(self, comm=MPI.COMM_WORLD, size=0, nnz=1, dtype=np.float64, submap=1, local=None):
        self._comm = comm
        self._size = size
        self._nnz = nnz
        self._dtype = dtype
        self._submap = submap
        self._local = local
        self._glob2loc = {}

        # our data is a 3D array of submap, pixel, values
        # we allocate this as a contiguous block
        if self._local is None:
            self.data = None
            self._nsub = 0
        else:
            self._nsub = len(self._local)
            for g in enumerate(self._local):
                self._glob2loc[g[1]] = g[0]
            if (self._submap * self._local.max()) > self._size:
                 raise RuntimeError("local submap indices out of range")
            self.data = np.zeros( (self._nsub * self._submap * self._nnz), order='C', dtype=self._dtype).reshape(self._nsub, self._submap, self._nnz)


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
    def local(self):
        return self._local

    @property
    def submap(self):
        return self._submap
    
    @property
    def nsubmap(self):
        return self._nsub


    def global_to_local(self, global):
        sm = np.floor_divide(global, self._submap)
        pix = np.mod(global, self._submap)
        f = (self._glob2loc[x] for x in sm)
        lsm = np.fromiter(f, np.int64, count=len(sm))
        return (lsm, pix)


    def duplicate(self):
        ret = DistPixels(comm=self._comm, size=self._size, nnz=self._nnz, dtype=self._dtype, submap=self._submap, local=self._local)
        if self.data is not None:
            ret.data = np.copy(self.data)
        return ret


    def read_healpix_fits(self, path):
        # For performance reasons, we can't use healpy to read this
        # map, since we want to read in a buffered way all maps and
        # Bcast.


        
        return


    def write_healpix_fits(self, path):
        raise RuntimeError('writing to healpix FITS not yet implemented')
        return
