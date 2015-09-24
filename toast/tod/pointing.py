# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import healpy as hp

import quaternionarray as qa

from ..operator import Operator
from ..dist import Comm, Data, Obs
from .tod import TOD


class OpPointingFake(Operator):
    """
    Operator which generates fake, uniform, pointing matrices.

    Args:
        nside (int): NSIDE resolution for Healpix NEST ordered intensity map.

    """

    def __init__(self, nside=64):
        self._nside = nside

        # We call the parent class constructor, which currently does nothing
        super().__init__()


    @property
    def nside(self):
        return self._nside


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

        zaxis = np.array([0,0,1], dtype=np.float64)

        for obs in data.obs:
            tod = obs.tod
            for det in tod.local_dets:
                pdata, pflags = tod.read_pntg(det, 0, tod.local_samples)
                dir = qa.rotate(pdata.reshape(-1, 4), zaxis)
                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2], nest=True)
                nnz = 1
                weights = np.ones(nnz * tod.local_samples, dtype=np.float64)
                tod.write_pmat(detector=det, local_start=0, pixels=pixels, weights=weights) 
        return


