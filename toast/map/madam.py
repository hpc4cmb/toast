# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import os
if 'PYTOAST_NOMPI' in os.environ.keys():
    from .. import fakempi as MPI
else:
    from mpi4py import MPI

import ctypes as ct
from ctypes.util import find_library

import unittest

import numpy as np
import numpy.ctypeslib as npc

from ..dist import Comm, Data
from ..operator import Operator
from ..tod import TOD
from ..tod import Interval


try:
    libmadam = ct.CDLL('libmadam.so')
except:
    libmadam = None

if libmadam is not None:
    libmadam.destripe.restype = None
    libmadam.destripe.argtypes = [
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
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]


class OpMadam(Operator):
    """
    Operator which passes data to libmadam for map-making.

    Args:
        params (dictionary): parameters to pass to madam.
        detweights (dictionary): individual noise weights to use for each
            detector.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        name (str): the name of the cache object (<name>_<detector>) to
            use for the detector timestream.  If None, use the TOD.
        purge (bool): if True, clear any cached data that is copied intersection
            the Madam buffers.
    """

    def __init__(self, params={}, detweights=None, pixels='pixels', weights='weights', name=None, flag_name=None, flag_mask=255, common_flag_name=None, common_flag_mask=255, apply_flags=True, purge=False):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        # madam uses time-based distribution
        self._timedist = True
        self._name = name
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._pixels = pixels
        self._weights = weights
        self._detw = detweights
        self._purge = purge
        self._apply_flags = apply_flags
        self._params = params


    @property
    def timedist(self):
        """
        (bool): Whether this operator requires data that time-distributed.
        """
        return self._timedist


    @property
    def available(self):
        """
        (bool): True if libmadam is found in the library search path.
        """
        return (libmadam is not None)
    

    def _dict2parstring(self, d):
        s = ''
        for key, value in d.items():
            s += '{} = {};'.format(key, value)
        return s


    def _dets2detstring(self, dets):
        s = ''
        for d in dets:
            s += '{};'.format(d)
        return s


    def exec(self, data):
        """
        Copy data to Madam-compatible buffers and make a map.

        Args:
            data (toast.Data): The distributed data.
        """
        if libmadam is None:
            raise RuntimeError("Cannot find libmadam")

        # the two-level pytoast communicator
        comm = data.comm

        # Madam only works with a data model where there is one observation
        # split among the processes, and where the data is distributed time-wise
        if len(data.obs) != 1:
            raise RuntimeError("Madam requires a single observation")

        tod = data.obs[0]['tod']
        if not tod.timedist:
            raise RuntimeError("Madam requires data to be distributed by time")

        # get the total list of intervals
        intervals = None
        if 'intervals' in data.obs[0].keys():
            intervals = data.obs[0]['intervals']
        if intervals is None:
            intervals = [Interval(start=0.0, stop=0.0, first=0, last=(tod.total_samples-1))]

        # get the noise object
        if 'noise' in data.obs[0].keys():
            nse = data.obs[0]['noise']
        else:
            nse = None

        todcomm = tod.mpicomm
        todfcomm = todcomm.py2f()

        # to get the number of Non-zero pointing weights per pixel,
        # we use the fact that for Madam, all processes have all detectors
        # for some slice of time.  So we can get this information from the
        # shape of the data from the first detector

        nnzname = "{}_{}".format(self._weights, tod.detectors[0])
        nnz = tod.cache.reference(nnzname).shape[1]

        ndet = len(tod.detectors)
        nlocal = tod.local_samples[1]

        parstring = self._dict2parstring(self._params)
        detstring = self._dets2detstring(tod.detectors)

        timestamps = tod.read_times(local_start=0, n=nlocal)

        # create madam-compatible buffers

        madam_signal = np.zeros(ndet * nlocal, dtype=np.float64)
        madam_pixels = np.zeros(ndet * nlocal, dtype=np.int64)
        madam_pixweights = np.zeros(ndet * nlocal * nnz, dtype=np.float64)

        for d in range(ndet):

            dslice = slice(d * nlocal, (d+1) * nlocal)
            dwslice = slice(d * nlocal * nnz, (d+1) * nlocal * nnz)
            
            # Get the signal.

            cachename = None
            if self._name is not None:
                cachename = "{}_{}".format(self._name, tod.detectors[d])
                signal = tod.cache.reference(cachename)
            else:
                signal = tod.read(detector=tod.detectors[d])
            madam_signal[dslice] = signal

            # Optionally get the flags, otherwise they are assumed to be have been applied
            # to the pixel numbers.

            if self._apply_flags:

                if self._flag_name is not None:
                    cacheflagname = "{}_{}".format(self._flag_name, tod.detectors[d])
                    detflags = tod.cache.reference(cacheflagname)
                    flags = (detflags & self._flag_mask) != 0
                    if self._common_flag_name is not None:
                        commonflags = tod.cache.reference(self._common_flag_name)
                        flags[(commonflags & self._common_flag_mask) != 0] = True
                else:
                    detflags, commonflags = tod.read_flags(detector=tod.detectors[d])
                    flags = np.logical_or((detflags & self._flag_mask) != 0, (commonflags & self._common_flag_mask) != 0)

            # get the pixels and weights from the cache

            pixelsname = "{}_{}".format(self._pixels, tod.detectors[d])
            weightsname = "{}_{}".format(self._weights, tod.detectors[d])
            pixels = tod.cache.reference(pixelsname)
            weights = tod.cache.reference(weightsname)

            if self._apply_flags:
                pixels = pixels.copy() # Don't change the cached pixel numbers
                pixels[flags] = -1

            madam_pixels[dslice] = pixels
            madam_pixweights[dwslice] = weights.flatten()

            if self._purge:
                tod.cache.clear(pattern=pixelsname)
                tod.cache.clear(pattern=weightsname)
                if self._name is not None:
                    tod.cache.clear(pattern=cachename)
                if self._flag_name is not None:
                    tod.cache.clear(pattern=cacheflagname)
                    
        if self._purge:
            if self._common_flag_name is not None:
                tod.cache.clear(pattern=self._common_flag_name)

        # The "pointing periods" we pass to madam are simply the intersection
        # of our local data and the list of valid intervals.

        local_bounds = [ (t.first - tod.local_samples[0]) if (t.first > tod.local_samples[0]) else 0 for t in intervals if (t.last >= tod.local_samples[0]) and (t.first < (tod.local_samples[0] + tod.local_samples[1])) ]
        
        nperiod = len(local_bounds)

        periods = np.zeros(nperiod, dtype=np.int64)
        for p in range(nperiod):
            periods[p] = int(local_bounds[p])

        # detweights is either a dictionary of weights specified at construction time,
        # or else we use uniform weighting.
        detw = {}
        if self._detw is None:
            for d in range(ndet):
                detw[tod.detectors[d]] = 1.0
        else:
            detw = self._detw

        detweights = np.zeros(ndet, dtype=np.float64)
        for d in range(ndet):
            detweights[d] = detw[tod.detectors[d]]

        if nse is not None:
            nse_psdfreqs = nse.freq
            npsdbin = len(nse_psdfreqs)
            psdfreqs = np.copy(nse_psdfreqs)

            npsd = np.ones(ndet, dtype=np.int64)
            npsdtot = np.sum(npsd)
            psdstarts = np.zeros(npsdtot, dtype=np.float64)

            npsdval = npsdbin * npsdtot
            psdvals = np.zeros(npsdval, dtype=np.float64)
            for d in range(ndet):
                psdvals[d*npsdbin:(d+1)*npsdbin] = nse.psd(tod.detectors[d])
        else:
            npsd = np.ones(ndet, dtype=np.int64)
            npsdtot = np.sum(npsd)
            psdstarts = np.zeros(npsdtot)
            npsdbin = 10
            fsample = 10.
            psdfreqs = np.arange(npsdbin) * fsample / npsdbin
            npsdval = npsdbin * npsdtot            
            psdvals = np.ones( npsdval )
            

        # destripe

        libmadam.destripe(todfcomm, parstring.encode(), ndet, detstring.encode(), detweights, nlocal, nnz, timestamps, madam_pixels, madam_pixweights, madam_signal, nperiod, periods, npsd, npsdtot, psdstarts, npsdbin, psdfreqs, npsdval, psdvals)

        return
