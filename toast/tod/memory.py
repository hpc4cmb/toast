# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..operator import Operator
from ..dist import Comm, Data, Obs
from .tod import TOD


class OpCopy(Operator):
    """
    Operator which copies input data into the native in-memory format.

    This passes through each observation and copies all data types into
    the base class implementation of those types (which store their data
    in memory).  It optionally changes the distribution scheme and
    redistributes the data when copying.

    Args:
        timedist (bool): if True, the output data will be distributed by 
                         time, otherwise by detector.  Data is shuffled
                         between processes if it is in a different
                         distribution.
    """

    def __init__(self, timedist=True):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._timedist = timedist

    @property
    def timedist(self):
        return self._timedist


    def _shuffle(self, data, flags, local_dets, local_offset, local_samples):
        raise NotImplementedError("re-mapping of data distribution not yet supported")
        return

    def _shuffle_times(self, stamps, local_dets, local_offset, local_samples):
        raise NotImplementedError("re-mapping of data distribution not yet supported")
        return

    def _shuffle_pointing(self, pixels, weights, local_dets, local_offset, local_samples):
        raise NotImplementedError("re-mapping of data distribution not yet supported")
        return


    def exec(self, indata):
        comm = indata.comm
        outdata = Data(comm)
        for inobs in indata.obs:
            tod = inobs.tod
            base = inobs.baselines
            nse = inobs.noise
            intrvl = inobs.intervals

            outtod = TOD(mpicomm=tod.mpicomm, timedist=self.timedist, 
                detectors=tod.detectors, flavors=tod.flavors, 
                samples=tod.total_samples)

            # FIXME:  add noise and baselines once implemented
            outbaselines = None
            outnoise = None
            
            if tod.timedist == self.timedist:
                # we have the same distribution, and just need
                # to read and write
                stamps = tod.read_times(local_start=0, n=tod.local_samples)
                outtod.write_times(local_start=0, stamps=stamps)
                for det in tod.local_dets:
                    pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples)
                    #print("copy input pdata, pflags have size: {}, {}".format(len(pdata), len(pflags)))
                    outtod.write_pntg(detector=det, local_start=0, data=pdata, flags=pflags)
                    for flv in tod.flavors:
                        data, flags = tod.read(detector=det, flavor=flv, local_start=0, n=tod.local_samples) 
                        outtod.write(detector=det, flavor=flv, local_start=0, data=data, flags=flags)
                    for name in tod.pointings:
                        pixels, weights = tod.read_pmat(name=name, detector=det, local_start=0, n=0)
                        outtod.write_pmat(name=name, detector=det, local_start=0, pixels=pixels, weights=weights)
            else:
                # we have to read our local piece and communicate
                stamps = tod.read_times(local_start=0, n=tod.local_samples)
                stamps = _shuffle_times(stamps, tod.local_dets, tod.local_offset, tod.local_samples)
                outtod.write_times(local_start=0, stamps=stamps)
                for det in tod.local_dets:
                    pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples)
                    pdata, pflags = _shuffle(pdata, pflags, tod.local_dets, tod.local_offset, tod.local_samples)
                    outtod.write_pntg(detector=det, local_start=0, data=pdata, flags=pflags)
                    for flv in tod.flavors:
                        data, flags = tod.read(detector=det, flavor=flv, local_start=0, n=tod.local_samples)
                        data, flags = _shuffle(data, flags, tod.local_dets, tod.local_offset, tod.local_samples)
                        outtod.write(detector=det, flavor=flv, local_start=0, data=data, flags=flags)
                    for name in tod.pointings:
                        pixels, weights = tod.read_pmat(name=name, detector=det, local_start=0, n=0)
                        pixels, weights = _shuffle_pointing(pixels, weights, tod.local_dets, tod.local_offset, tod.local_samples)
                        outtod.write_pmat(name=name, detector=det, local_start=0, pixels=pixels, weights=weights)

            outobs = Obs(tod=outtod, intervals=intrvl, baselines=outbaselines, noise = outnoise)

            outdata.obs.append(outobs)
        return outdata


