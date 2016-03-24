# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

import healpy as hp

import quaternionarray as qa

from ..operator import Operator
from ..dist import Comm, Data
from .tod import TOD



class OpPointingHpix(Operator):
    """
    Operator which generates I/Q/U healpix pointing weights.

    Given the individual detector pointing, this computes the pointing weights
    assuming that the detector is a linear polarizer followed by a total
    power measurement.

    Additional options include specifying a constant cross-polar response 
    and a rotating, perfect half-wave plate.

    Args:
        nside (int): NSIDE resolution for Healpix NEST ordered intensity map.
        nest (bool): if True, use NESTED ordering.
        mode (string): either "I" or "IQU"
        epsilon (dict): dictionary of cross-polar response per detector. A
            None value means epsilon is zero for all detectors.
        hwprate: if None, a constantly rotating HWP is not included.  Otherwise
            it is the rate (in RPM) of constant rotation.
        hwpstep: if None, then a stepped HWP is not included.  Otherwise, this
            is the step in degrees.
        hwpsteptime: The time in minutes between HWP steps.
        purge_pntg (bool): If True, clear the detector quaternion pointing
            after building the pointing matrix.
    """

    def __init__(self, nside=64, nest=False, mode='I', epsilon=None, hwprate=None, hwpstep=None, hwpsteptime=None, purge_pntg=False):
        self._nside = nside
        self._nest = nest
        self._mode = mode
        self._epsilon = epsilon
        self._purge = purge_pntg

        if (hwprate is not None) and (hwpstep is not None):
            raise RuntimeError("choose either continuously rotating or stepped HWP")

        if (hwpstep is not None) and (hwpsteptime is None):
            raise RuntimeError("for a stepped HWP, you must specify the time between steps")

        if hwprate is not None:
            # convert to radians / second
            self._hwprate = hwprate * 2.0 * np.pi / 60.0
        else:
            self._hwprate = None

        if hwpstep is not None:
            # convert to radians and seconds
            self._hwpstep = hwpstep * np.pi / 180.0
            self._hwpsteptime = hwpsteptime * 60.0
        else:
            self._hwpstep = None
            self._hwpsteptime = None

        # We call the parent class constructor, which currently does nothing
        super().__init__()


    @property
    def nside(self):
        return self._nside

    @property
    def nest(self):
        return self._nest

    @property
    def mode(self):
        return self._mode


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

        xaxis = np.array([1,0,0], dtype=np.float64)
        yaxis = np.array([0,1,0], dtype=np.float64)
        zaxis = np.array([0,0,1], dtype=np.float64)
        nullquat = np.array([0,0,0,1], dtype=np.float64)

        for obs in data.obs:
            tod = obs['tod']

            # compute effective sample rate

            times = tod.read_times(local_start=0, n=tod.local_samples[1])
            dt = np.mean(times[1:-1] - times[0:-2])
            rate = 1.0 / dt

            # generate HWP angles

            nsamp = tod.local_samples[1]
            first = tod.local_samples[0]
            hwpang = None

            if self._hwprate is not None:
                # continuous HWP
                # HWP increment per sample is: 
                # (hwprate / samplerate)
                hwpincr = self._hwprate / rate
                startang = np.fmod(first * hwpincr, 2*np.pi)
                hwpang = hwpincr * np.arange(nsamp, dtype=np.float64)
                hwpang += startang
            elif self._hwpstep is not None:
                # stepped HWP
                hwpang = np.ones(nsamp, dtype=np.float64)
                stepsamples = int(self._hwpsteptime * rate)
                wholesteps = int(first / stepsamples)
                remsamples = first - wholesteps * stepsamples
                curang = np.fmod(wholesteps * self._hwpstep, 2*np.pi)
                curoff = 0
                fill = remsamples
                while (curoff < nsamp):
                    if curoff + fill > nsamp:
                        fill = nsamp - curoff
                    hwpang[curoff:fill] *= curang
                    curang += self._hwpstep
                    curoff += fill
                    fill = stepsamples

            for det in tod.local_dets:

                eps = 0.0
                if self._epsilon is not None:
                    eps = self._epsilon[det]

                oneplus = 0.5 * (1.0 + eps)
                oneminus = 0.5 * (1.0 - eps)

                pdata, pflags = tod.read_pntg(detector=det, local_start=0, n=nsamp)
                pdata.reshape(-1,4)[pflags != 0,:] = nullquat

                dir = qa.rotate(pdata.reshape(-1, 4), np.tile(zaxis, nsamp).reshape(-1,3))

                pixels = hp.vec2pix(self._nside, dir[:,0], dir[:,1], dir[:,2], nest=self._nest)
                pixels[pflags != 0] = -1

                if self._mode == 'I':
                    
                    weights = np.ones(nsamp, dtype=np.float64)
                    weights *= oneplus
                    tod.write_pmat(detector=det, local_start=0, pixels=pixels, weights=weights)

                elif self._mode == 'IQU':

                    orient = qa.rotate(pdata.reshape(-1, 4), np.tile(xaxis, nsamp).reshape(-1,3))

                    by = orient[:,0] * dir[:,1] - orient[:,1] * dir[:,0]
                    bx = orient[:,0] * (-dir[:,2] * dir[:,0]) + orient[:,1] * (-dir[:,2] * dir[:,1]) + orient[:,2] * (dir[:,0] * dir[:,0] + dir[:,1] * dir[:,1])
                        
                    detang = np.arctan2(by, bx)
                    if hwpang is not None:
                        detang += hwpang
                        detang *= 4.0
                    else:
                        detang *= 2.0
                    cang = np.cos(detang)
                    sang = np.sin(detang)
                     
                    Ival = np.ones_like(cang)
                    Ival *= oneplus
                    Qval = cang
                    Qval *= oneminus
                    Uval = sang
                    Uval *= oneminus

                    weights = np.ravel(np.column_stack((Ival, Qval, Uval)))

                    tod.write_pmat(detector=det, local_start=0, pixels=pixels, weights=weights)
                else:
                    raise RuntimeError("invalid mode for healpix pointing")

                if self._purge:
                    tod.clear_pntg(detector=det)

        return

