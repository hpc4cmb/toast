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
from ..tod import Interval


class OpMadam(Operator):
    """
    Operator which passes data to libmadam for map-making.

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
        self._pmat = pmat
        if self._pmat is None:
            self._pmat = TOD.DEFAULT_FLAVOR
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

        # Madam only works with a data model where there is one observation
        # split among the processes, and where the data is distributed time-wise
        if len(data.obs) != 1:
            raise RuntimeError("Madam requires a single observation")

        tod = data.obs[0].tod
        if not tod.timedist:
            raise RuntimeError("Madam requires data to be distributed by time")

        intervals = data.obs[0].intervals

        todcomm = tod.mpicomm
        todfcomm = todcomm.py2f()

        # create madam-compatible buffers

        ndet = len(tod.detectors)
        nglobal = tod.total_samples
        nlocal = tod.local_samples
        nnz = tod.pmat_nnz(self._pmat)

        parstring = _dict2parstring(self._params)
        detstring = _dets2detstring(tod.detectors)

        timestamps = tod.read_times()

        signal = np.zeros(ndet * nlocal, dtype=np.float64)
        pixels = np.zeros(ndet * nlocal, dtype=np.int64)
        pixweights = np.zeros(ndet * nlocal * nnz, dtype=np.float64)

        for d in range(ndet):
            dslice = (d * nlocal, (d+1) * nlocal)
            dwslice = (d * nlocal * nnz, (d+1) * nlocal * nnz)
            signal[dslice] = tod.read(detector=d, flavor=self._flavor)
            pixels[dslice], pixweights[dwslice] = tod.read_pmat(name=self._pmat, detector=d)

        nperiod = len(intervals)
        periods = np.zeros(nperiod, dtype=np.int64)
        for p in range(nperiod):
            periods[p] = int(intervals[p].first)

        # use uniform white noise PSDs for now...

        detweights = np.ones(ndet, dtype=np.float64)

        npsd = np.ones(ndet, dtype=np.int64)
        npsdtot = np.sum(npsd)
        psdstarts = np.zeros(npsdtot, dtype=np.float64)
        npsdbin = 1
        npsdval = npsdbin * npsdtot
        psdfreqs = np.zeros(npsdtot, dtype=np.float64)
        psdvals = np.ones(npsdval, dtype=np.float64)

        # destripe

        OperatorMadam.lib_handle.destripe(
            fcomm,
            parstring,
            ndet,
            detstring,
            detweights,
            nlocal,
            nnz,
            timestamps,
            pixels,
            pixweights,
            signal,
            nperiod,
            periods,
            npsd,
            npsdtot,
            psdstarts,
            npsdbin,
            psdfreqs,
            npsdval,
            psdvals
        )

        return
