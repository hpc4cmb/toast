# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import Comm, Data, Obs
from ..tod import TOD
from .memory import Operator


class OperatorMadam(Operator):
    """
    Operator which passes data to libmadam for map-making.

    This passes through each observation and copies all data types into
    the base class implementation of those types (which store their data
    in memory).  It optionally changes the distribution scheme and
    redistributes the data when copying.

    Args:
        params (dictionary): parameters to pass to madam.
    """

    def __init__(self, flavor=None, params={}):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        # madam uses time-based distribution
        self._timedist = True
        self._flavor = flavor
        if self._flavor is None:
            self._flavor = TOD.DEFAULT_FLAVOR

    @property
    def timedist(self):
        return self._timedist

    def exec(self, tdata):
        comm = tdata.comm
        world = comm.comm_world

        # do these need to be pre-allocated over the global
        # communicator?
        map = None
        binned = None
        hits = None

        # create a madam-compatible TOD/pointing buffer that
        # is distributed over the world communicator.
        # This probably requires looping the same way over all
        # observations and find the total number of samples
        # (by calling TOD.total_samples())


        # this loop is happening over the group-communicator
        # assigned list of observations

        for obs in tdata.obs:
            tod = obs.tod
            nse = obs.noise
            
            for det in tod.local_dets:
                pdata, pflags = tod.read_pntg(det, 0, tod.local_samples[1])
                data, flags = tod.read(det, self._flavor, 0, tod.local_samples[1])
                # Each process in the group communicator now has a piece of the data.
                # Distribute this to a madam-compatible TOD/pointing buffer that
                # is distributed over the global communicator.

        # call libmadam!


        return map, binned, hits


