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

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, samples=0):

        self.mpicomm = mpicomm
        self.timedist = timedist
        self.detectors = []
        if detectors != None:
            self.detectors = detectors
        self.samples = samples
        self.ndata = 4 * self.samples

        (self.dist_dets, self.dist_samples) = distribute_det_samples(self.mpicomm, self.timedist, self.detectors, self.samples)

        self.dist_ndata = (4*self.dist_samples[0], 4*self.dist_samples[1])

        self.data = {}
        for det in self.detectors:
            self.data[det] = np.zeros(self.dist_ndata[1], dtype=np.float64)
        self.flags = {}
        for det in self.detectors:
            self.flags[det] = np.zeros(self.dist_samples[1], dtype=np.uint8)


    def _get(self, detector, start, n):
        return (self.data[detector][4*start:4*(start+n)], self.flags[detector][start:start+n])


    def _put(self, detector, start, data, flags):
        n = flags.shape[0]
        self.data[detector][4*start:4*(start+n)] = np.copy(data)
        self.flags[detector][start:start+n] = np.copy(flags)
        return


    def valid_dets(self):
        return self.detectors


    def is_timedist(self):
        return self.timedist


    def total_samples(self):
        return self.samples


    def local_samples(self):
        return self.dist_samples


    def local_dets(self):
        return self.dist_dets


    def mpicomm(self):
        return self.mpicomm


    def read(self, detector=None, start=0, n=0):
        if detector not in self.valid_dets():
            raise ValueError('detector {} not found'.format(detector))
        if (start < 0) or (start + n > self.nsamp()):
            raise ValueError('sample range {} - {} is invalid'.format(start, start+n-1))
        return self._get(detector, start, n)


    def write(self, detector=None, start=0, data=None, flags=None):
        if detector not in self.valid_dets():
            raise ValueError('detector {} not found'.format(detector))
        if data is None or flags is None:
            raise ValueError('both data and flags must be specified')
        if (data.shape[0] != 4*flags.shape[0]):
            raise ValueError('data and flags arrays have inconsistent length')
        if (start < 0) or (start + flags.shape[0] > self.nsamp()):
            raise ValueError('sample range {} - {} is invalid'.format(start, start+flags.shape[0]-1))
        self._put(detector, start, data, flags)
        return


