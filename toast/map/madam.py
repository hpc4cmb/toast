# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import ctypes as ct
from ctypes.util import find_library

import unittest

import numpy as np
import numpy.ctypeslib as npc

from ..dist import Comm, Data, Obs
from ..operator import Operator
from ..tod import TOD


class OpMadam(Operator):
    """
    Operator which passes data to libmadam for map-making.

    This passes through each observation and copies all data types into
    the base class implementation of those types (which store their data
    in memory).  It optionally changes the distribution scheme and
    redistributes the data when copying.

    Args:
        params (dictionary): parameters to pass to madam.
    """

    # We store the shared library handle as a class
    # attribute, so that we only ever dlopen the library
    # once.

    lib_path = ""
    lib_handle = None

    def __init__(self, flavor=None, pmat=None, params={}):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        # madam uses time-based distribution
        self._timedist = True
        self._flavor = flavor
        if self._flavor is None:
            self._flavor = TOD.DEFAULT_FLAVOR
        self._params = params

        # dlopen the madam library, if not done already
        if OperatorMadam.lib_handle is None:
            OperatorMadam.lib_path = find_library('libmadam')
            OperatorMadam.lib_handle = ct.CDLL(OperatorMadam.lib_path, mode=ct.RTLD_GLOBAL)
            OperatorMadam.lib_handle.destripe.restype = None
            OperatorMadam.lib_handle.destripe.argtypes = [
                ct.c_int,
                ct.c_char_p,
                ct.c_long,
                ct.c_char_p,
                npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                ct.c_long,
                ct.c_long,
                npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
                npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                ct.c_long,
                npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
                npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
                ct.c_long,
                npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                ct.c_long,
                npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                ct.c_long,
                npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
            ]


    @property
    def timedist(self):
        return self._timedist


    def _dict2parstring(d):
        s = ''
        for key, value in d.items():
            s += '{} = {};'.format(key, value)
        return s


    def _dets2detstring(dets):
        s = ''
        for d in dets:
            s += '{};'.format(d)
        return s


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
        fworld = cworld.py2f()

        # Determine the global list of detectors.

        global_dets = []
        global_samples = 0
        group_dets = []
        group_samples = 0

        # For all obs assigned to our group,
        # get the superset of all detectors.
        # Also accumulate our group's total samples.
        for obs in data.obs:
            tod = obs.tod
            for det in tod.detectors:
                if det not in group_dets:
                    group_dets.extend(det)
            group_samples += tod.total_samples

        # The root processes in each group build the 
        # global list of detectors and samples and 
        # broadcast to their groups.
        if cgroup.rank == 0:
            crank.Allreduce(group_samples, global_samples, MPI.SUM)
        cgroup.Bcast(global_samples, root=0)

        if cworld.rank == 0:
            global_dets = group_dets

        for g in range(comm.ngroups):
            if g > 0:
                if cworld.rank == 0:
                    s = crank.recv(source=g, tag=g)
                    for det in s:
                        if det not in global_dets:
                            global_dets.extend(det)
                else:
                    if comm.group == g:
                        if cgroup.rank == 0:
                            crank.send(group_dets, dest=0, tag=g)

        global_dets = crank.bcast(global_dets, root=0)
        global_dets = cgroup.bcast(global_dets, root=0)

        # create a madam-compatible TOD/pointing buffer that
        # is distributed over the world communicator.



        for obs in tdata.obs:
            tod = obs.tod
            nse = obs.noise
            
            for det in tod.local_dets:
                pdata, pflags = tod.read_pntg(det, 0, tod.local_samples[1])
                data, flags = tod.read(det, self._flavor, 0, tod.local_samples[1])
                # Each process in the group communicator now has a piece of the data.
                # Distribute this to a madam-compatible TOD/pointing buffer that
                # is distributed over the global communicator.

        # destripe
        OperatorMadam.lib_handle.destripe()

        return
