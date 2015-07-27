# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import distribute_det_samples


class Pointing(object):
    """
    Base class for an object that provides detector pointing for 
    an observation.

    Each Pointing class has one or more detectors, and this class
    provides pointing quaternions and flags for each detector.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
            detector.
        detectors (list): list of names to use for the detectors.
        samples (int): pre-initialize the storage with this number of samples.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, samples=0, sizes=None):

        self._mpicomm = mpicomm
        self._timedist = timedist
        self._dets = []
        if detectors is not None:
            self._dets = detectors
        self._nsamp = samples
        self._ndata = 4 * self._nsamp
        self._sizes = sizes

        (self._dist_dets, self._dist_samples) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp, sizes=self._sizes)

        self._dist_ndata = (4*self._dist_samples[0], 4*self._dist_samples[1])

        self.data = {}
        for det in self._dist_dets:
            self.data[det] = np.zeros(self._dist_ndata[1], dtype=np.float64)

        self.flags = {}
        for det in self._dist_dets:
            self.flags[det] = np.zeros(self._dist_ndata[1], dtype=np.uint8)


    @property
    def detectors(self):
        return self._dets

    @property
    def timedist(self):
        return self._timedist

    @property
    def total_samples(self):
        return self._nsamp

    @property
    def local_samples(self):
        return self._dist_samples

    @property
    def local_dets(self):
        return self._dist_dets

    @property
    def mpicomm(self):
        return self._mpicomm


    def _get(self, detector, start, n):
        return (self.data[detector][4*start:4*(start+n)], self.flags[detector][start:start+n])


    def _put(self, detector, start, data, flags):
        n = flags.shape[0]
        self.data[detector][4*start:4*(start+n)] = np.copy(data)
        self.flags[detector][start:(start+n)] = np.copy(flags)
        return


    def read(self, detector=None, local_start=0, n=0):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get(detector, local_start, n)


    def write(self, detector=None, local_start=0, data=None, flags=None):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape[0] != 4 * flags.shape[0]:
            raise ValueError('data and flags arrays must be the same number of samples')
        if (local_start < 0) or (local_start + flags.shape[0] > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+data.shape[0]-1))
        self._put(detector, local_start, data, flags)
        return


class PointingSimSingle(object):
    """
    Provide a single-detector (essentially boresight) pointing
    for use in testing.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        samples (int): pre-initialize the storage with this number of samples.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, samples=0):

        # We call the parent class constructor to set the MPI communicator and
        # distribution type, but we do NOT pass the detector list, as this 
        # would allocate memory for the data buffer of the base class.
        super().__init__(mpicomm=mpicomm, timedist=True, detectors=None, flavors=None, samples=0)

        # we only provide one detector
        self._dets = ['default']

        self._nsamp = samples
        self._ndata = 4 * self._nsamp

        (self._dist_dets, self._dist_samples) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp)

        self._dist_ndata = (4*self._dist_samples[0], 4*self._dist_samples[1])

        self._theta_steps = 180
        self._phi_steps = 360


    def _get(self, detector, start, n):
        # compute the absolute sample offset
        start_abs = self._dist_samples[0] + start

        # compute the position on the sphere, spiralling from North
        # pole to South.  obviously the pointings will be densest at
        # the poles.
        start_theta = int(start_abs / self._theta_steps)
        start_phi = int(start_abs / self._phi_steps)

        # ... in progress ...

        # convert to quaternions
        # FIXME: use our quaternion library once implemented

        data = np.zeros(n, dtype=np.float64)
        flags = np.zeros(n, dtype=np.uint8)

        return (data, flags)


    def _put(self, detector, start, data, flags):
        raise RuntimeError('cannot write data to simulated pointing class')
        return


