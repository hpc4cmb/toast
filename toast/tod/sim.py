# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import healpy as hp

import quaternionarray as qa

from ..dist import distribute_det_samples

from .tod import TOD

from ..operator import Operator


class TODFake(TOD):
    """
    Provide a simple generator of fake detector pointing.

    Detector focalplane offsets are specified as a dictionary of
    quaternion tuples.  The boresight pointing is a simple looping 
    over HealPix ring ordered pixel centers.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        detectors (dictionary): each key is the detector name, and each value
                  is a quaternion tuple.
        samples (int): maximum allowed samples.
        firsttime (float): starting time of data.
        rate (float): sample rate in Hz.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, detectors=None, samples=0, firsttime=0.0, rate=100.0):
        
        super().__init__(mpicomm=mpicomm, timedist=True, detectors=detectors.keys(), flavors=None, samples=samples)

        self._fp = detectors        
        self._firsttime = firsttime
        self._rate = rate


    def _get(self, detector, flavor, start, n):
        # This class just returns data streams of zeros
        return ( np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.uint8) )


    def _put(self, detector, flavor, start, data, flags):
        raise RuntimeError('cannot write data to simulated data streams')
        return


    def _get_times(self, start, n):
        start_abs = self.local_offset + start
        start_time = self.firsttime + float(start_abs) / self.rate
        stop_time = start_time + float(n) / self.rate
        stamps = np.linspace(start_time, stop_time, num=n, endpoint=False, dtype=np.float64)
        return stamps


    def _put_times(self, start, stamps):
        raise RuntimeError('cannot write timestamps to simulated data streams')
        return


    def _get_pntg(self, detector, start, n):
        # compute the absolute sample offset
        start_abs = self.local_offset + start


        data = np.zeros(4*n, dtype=np.float64)
        flags = np.zeros(n, dtype=np.uint8)

        return (data, flags)


    def _put_pntg(self, detector, start, data, flags):
        raise RuntimeError('cannot write data to simulated pointing')
        return


class OpSimNoise(Operator):
    """
    Operator which generates noise timestreams and accumulates that data
    to a particular timestream flavor.

    This passes through each observation and ...

    Args:
        
    """

    def __init__(self, flavor=None):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._flavor = flavor

        self._rngstream = rngstream
        self._seeds = {}
        for det in enumerate(self.detectors):
            self._seeds[det[1]] = det[0] 
        self._rms = rms

    @property
    def timedist(self):
        return self._timedist

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
                for det in tod.local_dets:
                    pdata, pflags = tod.read_pntg(det, 0, tod.local_samples[1])
                    print("copy input pdata, pflags have size: {}, {}".format(len(pdata), len(pflags)))
                    outtod.write_pntg(det, 0, pdata, pflags)
                    for flv in tod.flavors:
                        data, flags = tod.read(det, flv, 0, tod.local_samples[1]) 
                        outtod.write(det, flv, 0, data, flags)
            else:
                # we have to read our local piece and communicate
                for det in tod.local_dets:
                    pdata, pflags = tod.read_pntg(det, 0, tod.local_samples[1])
                    pdata, pflags = _shuffle(pdata, pflags, tod.local_dets, tod.local_samples)
                    outtod.write_pntg(det, 0, pdata, pflags)
                    for flv in tod.flavors:
                        data, flags = tod.read(det, flv, 0, tod.local_samples[1])
                        data, flags = _shuffle(data, flags, tod.local_dets, tod.local_samples)
                        outstr.write(det, flv, 0, data, flags)

            outobs = Obs(tod=outtod, intervals=intrvl, baselines=outbaselines, noise = outnoise)

            outdata.obs.append(outobs)
        return outdata



        # Setting the seed like this does NOT guarantee uncorrelated
        # results from the generator.  This is just a place holder until
        # the streamed rng is implemented.
        np.random.seed(self.seeds[detector])
        trash = np.random.normal(loc=0.0, scale=self.rms, size=(n-start))
        return ( np.random.normal(loc=0.0, scale=self.rms, size=n), np.zeros(n, dtype=np.uint8) )



