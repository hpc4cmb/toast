# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import healpy as hp

from ..dist import Comm, Data
from ..operator import Operator
from ..tod import TOD


class OpLocalPixels(Operator):
    """
    Operator which computes the set of locally hit pixels.

    Args:
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
    """

    def __init__(self, pixels='pixels'):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        # madam uses time-based distribution
        self._timedist = True
        self._pixels = pixels


    @property
    def timedist(self):
        """
        (bool): Whether this operator requires data that time-distributed.
        """
        return self._timedist


    def exec(self, data):
        """
        Iterate over all observations and detectors and compute local pixels.

        Args:
            data (toast.Data): The distributed data.

        Returns:
            (array): An array of the locally hit pixel indices.
        """
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
                pixelsname = "{}_{}".format(self._pixels, det)
                pixels = tod.cache.reference(pixelsname)
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
    def __init__(self, comm=MPI.COMM_WORLD, size=0, nnz=1, dtype=np.float64, submap=None, local=None):
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
        """
        (mpi4py.MPI.Comm): The MPI communicator used.
        """
        return self._comm

    @property
    def size(self):
        """
        (int): The global number of pixels.
        """
        return self._size

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
    def local(self):
        """
        (array): The list of local submaps or None if process has no data.
        """
        return self._local

    @property
    def submap(self):
        """
        (int): The number of pixels in each submap.
        """
        return self._submap
    
    @property
    def nsubmap(self):
        """
        (int): The number of submaps stored on this process.
        """
        return self._nsub


    def global_to_local(self, gl):
        """
        Convert global pixel indices into the local submap and pixel.

        Args:
            gl (int): The global pixel number.

        Returns:
            A tuple containing the local submap index (int) and the
            pixel index local to that submap (int).
        """
        safe_gl = np.zeros_like(gl)
        good = (gl >= 0)
        bad = (gl < 0)
        safe_gl[good] = gl[good]
        sm = np.floor_divide(safe_gl, self._submap)
        pix = np.mod(safe_gl, self._submap)
        pix[bad] = -1
        f = (self._glob2loc[x] for x in sm)
        lsm = np.fromiter(f, np.int64, count=len(sm))
        return (lsm, pix)


    def duplicate(self):
        """
        Perform a deep copy of the distributed data.

        Returns:
            (DistPixels): A copy of the object.
        """
        ret = DistPixels(comm=self._comm, size=self._size, nnz=self._nnz, dtype=self._dtype, submap=self._submap, local=self._local)
        if self.data is not None:
            ret.data = np.copy(self.data)
        return ret


    def read_healpix_fits(self, path, buffer=5000000):

        elems = hp.read_map(path, dtype=self._dtype, memmap=True)

        nblock = len(elems)
        nnz = int( ( (np.sqrt(8*nblock) - 1) / 2 ) + 0.5 )

        # with memmap enabled, we read the underlying file
        # in chunks based on the internal FITS blocksize (2880 bytes)

        nsmbuf = buffer 


        return


    def write_healpix_fits(self, path):
        # healpy will be slow, since it must read the whole file NNZ
        # times as it grabs each column.  The long term solution is to
        # use a better format like HDF5.

        return
