# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np

from ..dist import distribute_det_samples


class Streams(object):
    """
    Base class for an object that provides a collection of streams.

    Each Streams class has one or more detectors, and each detctor
    might have different flavors of data and flags.

    Attributes:
        DEFAULT_FLAVOR (string): the name of the default flavor which always exists.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
            detector.
        detectors (list): list of names to use for the detectors.
        flavors (list): list of *EXTRA* flavors to use (beyond the default).
        samples (int): pre-initialize the storage with this number of samples.
    """

    DEFAULT_FLAVOR = 'default'

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, flavors=None, samples=0):

        self.mpicomm = mpicomm
        self.timedist = timedist
        self.detectors = []
        if detectors != None:
            self.detectors = detectors
        self.flavors = [self.DEFAULT_FLAVOR]
        if flavors != None:
            self.flavors.extend(flavors)
        self.samples = samples

        (self.dist_dets, self.dist_samples) = distribute_det_samples(self.mpicomm, self.timedist, self.detectors, self.samples)

        self.data = {}
        for det in self.dist_dets:
            self.data[det] = {}
            for flv in self.flavors:
                self.data[det][flv] = np.zeros(self.dist_samples[1], dtype=np.float64)
        self.flags = {}
        for det in self.dist_dets:
            self.flags[det] = {}
            for flv in self.flavors:
                self.flags[det][flv] = np.zeros(self.dist_samples[1], dtype=np.uint8)


    def _get(self, detector, flavor, start, n):
        return (self.data[detector][flavor][start:start+n], self.flags[detector][flavor][start:start+n])


    def _put(self, detector, flavor, start, data, flags):
        n = data.shape[0]
        self.data[detector][flavor][start:start+n] = np.copy(data)
        self.flags[detector][flavor][start:start+n] = np.copy(flags)
        return


    def valid_dets(self):
        return self.detectors


    def valid_flavors(self):
        return self.flavors


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


    def read(self, detector=None, flavor='default', local_start=0, n=0):
        if detector not in self.local_dets():
            raise ValueError('detector {} not found'.format(detector))
        if flavor not in self.valid_flavors():
            raise ValueError('flavor {} not found'.format(flavor))
        if (local_start < 0) or (local_start + n > self.local_samples()[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get(detector, flavor, local_start, n)


    def write(self, detector=None, flavor='default', local_start=0, data=None, flags=None):
        if detector not in self.local_dets():
            raise ValueError('detector {} not found'.format(detector))
        if flavor not in self.valid_flavors():
            raise ValueError('flavor {} not found'.format(flavor))
        if data is None or flags is None:
            raise ValueError('both data and flags must be specified')
        if data.shape != flags.shape:
            raise ValueError('data and flags arrays must be the same length')
        if (local_start < 0) or (local_start + data.shape[0] > self.local_samples()[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+data.shape[0]-1))
        self._put(detector, flavor, local_start, data, flags)
        return



class StreamsWhiteNoise(Streams):
    """
    Provide white noise streams.

    Args:
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which the data is distributed.
        timedist (bool): if True, the data is distributed by time, otherwise by
                  detector.
        detectors (list): list of names to use for the detectors.
        rms (float): RMS of the white noise.
        samples (int): maximum allowed samples.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, rms=1.0, samples=0, rngstream=0):
        
        # We call the parent class constructor to set the MPI communicator and
        # distribution type, but we do NOT pass the detector list, as this 
        # would allocate memory for the data buffer in the base class.
        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=None, flavors=None, samples=0)

        if detectors is None:
            raise ValueError('you must specify a list of detectors')
        self.detectors = detectors

        self.rngstream = rngstream
        self.seeds = {}
        for dets in enumerate(self.detectors):
            self.seeds[dets[1]] = dets[0] 
        self.flavors = [self.DEFAULT_FLAVOR]
        self.samples = samples
        self.rms = rms

        (self.dist_dets, self.dist_samples) = distribute_det_samples(self.mpicomm, self.timedist, self.detectors, self.samples)


    def _get(self, detector, flavor, start, n):
        # Setting the seed like this does NOT guarantee uncorrelated
        # results from the generator.  This is just a place holder until
        # the streamed rng is implemented.
        np.random.seed(self.seeds[detector])
        trash = np.random.normal(loc=0.0, scale=self.rms, size=(n-start))
        return ( np.random.normal(loc=0.0, scale=self.rms, size=n), np.zeros(n, dtype=np.uint8) )


    def _put(self, detector, flavor, start, data, flags):
        raise RuntimeError('cannot write data to simulated white noise streams')
        return

