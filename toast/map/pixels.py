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

        return local
