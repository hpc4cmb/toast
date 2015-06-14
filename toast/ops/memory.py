# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import Comm, Dist, Obs
from ..tod.pointing import Pointing
from ..tod.streams import Streams


class Operator(object):
    """
    Base class for an operator that acts on collections of observations.

    An operator takes as input a toast.dist.Dist object and returns a
    new instance of the same size.  For each observation in the distributed
    data, an operator may pass some data types forward unchanged, or it may
    replace or modify data.

    Currently this class does nothing, but may in the future...

    Args:
        None
    """

    def __init__(self):
        pass

    def exec(self, indist):
        return indist


class OperatorCopy(Operator):
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


    def _shuffle(self, data, flags, local_dets, local_samples):
        raise NotImplementedError("re-mapping of data distribution not yet supported")
        return


    def exec(self, indist):
        comm = indist.comm
        outdist = Dist(comm)
        for inobs in indist.obs:
            str = inobs.streams
            pntg = inobs.pointing
            base = inobs.baselines
            nse = inobs.noise

            if str.timedist != pntg.timedist:
                raise RuntimeError("pointing and streams objects have different distributions!")

            outstr = Streams(mpicomm=str.mpicomm, timedist=self.timedist, 
                detectors=str.detectors, flavors=str.flavors, 
                samples=str.total_samples)
            
            outpntg = Pointing(mpicomm=pntg.mpicomm, timedist=self.timedist,
                detectors=pntg.detectors, samples=pntg.total_samples)

            # FIXME:  add noise and baselines once implemented
            outbaselines = None
            outnoise = None
            
            if str.timedist == self.timedist:
                # we have the same distribution, and just need
                # to read and write
                for det in str.local_dets:
                    pdata, pflags = pntg.read(det, 0, pntg.local_samples[1])
                    outpntg.write(det, 0, pdata, pflags)
                    for flv in str.flavors:
                        data, flags = str.read(det, flv, 0, str.local_samples[1]) 
                        outstr.write(det, flv, 0, data, flags)
            else:
                # we have to read our local piece and communicate
                for det in str.local_dets:
                    pdata, pflags = pntg.read(det, 0, pntg.local_samples[1])
                    pdata, pflags = _shuffle(pdata, pflags, str.local_dets, str.local_samples)
                    outpntg.write(det, 0, pdata, pflags)
                    for flv in str.flavors:
                        data, flags = str.read(det, flv, 0, str.local_samples[1])
                        data, flags = _shuffle(data, flags, str.local_dets, str.local_samples)
                        outstr.write(det, flv, 0, data, flags)

            outobs = Obs(mpicomm=inobs.mpicomm, streams=outstr, pointing=outpntg, 
                baselines=outbaselines, noise = outnoise)

            outdist.obs.append(outobs)
        return outdist


