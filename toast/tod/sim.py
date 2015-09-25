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

    Detector focalplane offsets are specified as a dictionary of 4-element
    ndarrays.  The boresight pointing is a simple looping over HealPix 
    ring ordered pixel centers.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        detectors (dictionary): each key is the detector name, and each value
                  is a quaternion tuple.
        samples (int): maximum allowed samples.
        firsttime (float): starting time of data.
        rate (float): sample rate in Hz.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, detectors=None, samples=0, firsttime=0.0, rate=100.0, nside=512):
        if detectors is None:
            self._fp = {TOD.DEFAULT_FLAVOR : np.array([0.0, 0.0, 1.0, 0.0])}
        else:
            self._fp = detectors

        self._detlist = sorted(list(self._fp.keys()))
        
        super().__init__(mpicomm=mpicomm, timedist=True, detectors=self._detlist, flavors=None, samples=samples)

        self._firsttime = firsttime
        self._rate = rate
        self._nside = nside
        self._npix = 12 * self._nside * self._nside


    def _get(self, detector, flavor, start, n):
        # This class just returns data streams of zeros
        return ( np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.uint8) )


    def _put(self, detector, flavor, start, data, flags):
        raise RuntimeError('cannot write data to simulated data streams')
        return


    def _get_times(self, start, n):
        start_abs = self.local_offset + start
        start_time = self._firsttime + float(start_abs) / self._rate
        stop_time = start_time + float(n) / self._rate
        stamps = np.linspace(start_time, stop_time, num=n, endpoint=False, dtype=np.float64)
        return stamps


    def _put_times(self, start, stamps):
        raise RuntimeError('cannot write timestamps to simulated data streams')
        return


    def _get_pntg(self, detector, start, n):
        # compute the absolute sample offset
        start_abs = self.local_offset + start

        detquat = np.asarray(self._fp[detector])

        # pixel offset
        start_pix = int(start_abs % self._npix)
        pixels = np.linspace(start_pix, start_pix + n, num=n, endpoint=False)
        pixels = np.mod(pixels, self._npix*np.ones(n, dtype=np.int64)).astype(np.int64)

        # the result of this is normalized
        x, y, z = hp.pix2vec(self._nside, pixels, nest=False)

        # z axis is obviously normalized
        zaxis = np.array([0,0,1], dtype=np.float64)
        ztiled = np.tile(zaxis, x.shape[0]).reshape(-1,3)

        # ... so dir is already normalized
        dir = np.ravel(np.column_stack((x, y, z))).reshape(-1,3)

        # get the rotation axis
        v = np.cross(ztiled, dir)
        v = v / np.sqrt(np.sum(v * v, axis=1)).reshape(-1,1)

        # this is the vector-wise dot product
        zdot = np.sum(ztiled * dir, axis=1).reshape(-1,1)
        ang = 0.5 * np.arccos(zdot)

        # angle element
        s = np.cos(ang)

        # axis
        v *= np.sin(ang)

        # build the un-normalized quaternion
        boresight = np.concatenate((v, s), axis=1)

        boresight = qa.norm(boresight)

        # boredir = qa.rotate(boresight, zaxis)
        # boredir = boredir / np.sum(boredir * boredir, axis=1).reshape(-1,1)

        # check = hp.vec2pix(self._nside, boredir[:,0], boredir[:,1], boredir[:,2], nest=False)
        # if not np.array_equal(pixels, check):
        #     print(list(enumerate(zip(dir,boredir))))
        #     print(pixels)
        #     print(check)
        #     raise RuntimeError('FAIL on TODFake')

        flags = np.zeros(n, dtype=np.uint8)
        data = qa.mult(boresight, detquat).flatten()

        return (data, flags)


    def _put_pntg(self, detector, start, data, flags):
        raise RuntimeError('cannot write data to simulated pointing')
        return


class OpSimGradient(Operator):
    """
    Operator which generates fake sky signal as a gradient between the poles
    and accumulates this.

    This passes through each observation and ...

    Args:
        
    """

    def __init__(self, nside=512, flavor=None, min=-100.0, max=100.0):
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        self._nside = nside
        self._flavor = flavor
        self._min = min
        self._max = max

    def exec(self, data):
        comm = data.comm

        zaxis = np.array([0,0,1], dtype=np.float64)
        range = self._max - self._min

        for obs in data.obs:
            tod = obs.tod
            base = obs.baselines
            nse = obs.noise
            intrvl = obs.intervals

            for det in tod.local_dets:
                pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=tod.local_samples)
                dir = qa.rotate(pdata.reshape(-1, 4), zaxis)
                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2], nest=False)
                x, y, z = hp.pix2vec(self._nside, pixels, nest=False)
                z += 1.0
                z *= 0.5
                z *= range
                z += self._min
                data, flags = tod.read(detector=det, flavor=self._flavor, local_start=0, n=tod.local_samples)
                data += z
                tod.write(detector=det, flavor=self._flavor, local_start=0, data=data, flags=flags)

        return



# class OpSimNoise(Operator):
#     """
#     Operator which generates noise timestreams and accumulates that data
#     to a particular timestream flavor.

#     This passes through each observation and ...

#     Args:
        
#     """

#     def __init__(self, flavor=None, rms=1.0, ):
        
#         # We call the parent class constructor, which currently does nothing
#         super().__init__()
#         self._flavor = flavor

#         self._rngstream = rngstream
#         self._seeds = {}
#         for det in enumerate(self.detectors):
#             self._seeds[det[1]] = det[0] 
#         self._rms = rms

#     @property
#     def timedist(self):
#         return self._timedist

#     def exec(self, indata):
#         comm = indata.comm
#         outdata = Data(comm)
#         for inobs in indata.obs:
#             tod = inobs.tod
#             base = inobs.baselines
#             nse = inobs.noise
#             intrvl = inobs.intervals

#             outtod = TOD(mpicomm=tod.mpicomm, timedist=self.timedist, 
#                 detectors=tod.detectors, flavors=tod.flavors, 
#                 samples=tod.total_samples)

#             # FIXME:  add noise and baselines once implemented
#             outbaselines = None
#             outnoise = None
            
#             if tod.timedist == self.timedist:
#                 # we have the same distribution, and just need
#                 # to read and write
#                 for det in tod.local_dets:
#                     pdata, pflags = tod.read_pntg(det, 0, tod.local_samples[1])
#                     print("copy input pdata, pflags have size: {}, {}".format(len(pdata), len(pflags)))
#                     outtod.write_pntg(det, 0, pdata, pflags)
#                     for flv in tod.flavors:
#                         data, flags = tod.read(det, flv, 0, tod.local_samples[1]) 
#                         outtod.write(det, flv, 0, data, flags)
#             else:
#                 # we have to read our local piece and communicate
#                 for det in tod.local_dets:
#                     pdata, pflags = tod.read_pntg(det, 0, tod.local_samples[1])
#                     pdata, pflags = _shuffle(pdata, pflags, tod.local_dets, tod.local_samples)
#                     outtod.write_pntg(det, 0, pdata, pflags)
#                     for flv in tod.flavors:
#                         data, flags = tod.read(det, flv, 0, tod.local_samples[1])
#                         data, flags = _shuffle(data, flags, tod.local_dets, tod.local_samples)
#                         outstr.write(det, flv, 0, data, flags)

#             outobs = Obs(tod=outtod, intervals=intrvl, baselines=outbaselines, noise = outnoise)

#             outdata.obs.append(outobs)
#         return outdata



#         # Setting the seed like this does NOT guarantee uncorrelated
#         # results from the generator.  This is just a place holder until
#         # the streamed rng is implemented.
#         np.random.seed(self.seeds[detector])
#         trash = np.random.normal(loc=0.0, scale=self.rms, size=(n-start))
#         return ( np.random.normal(loc=0.0, scale=self.rms, size=n), np.zeros(n, dtype=np.uint8) )



