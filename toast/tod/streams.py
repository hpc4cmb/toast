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

    Each Streams class has one or more detectors, and each detector
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

        self._mpicomm = mpicomm
        self._timedist = timedist

        self._dets = []
        if detectors is not None:
            self._dets = detectors
        self._flavors = [self.DEFAULT_FLAVOR]
        if flavors is not None:
            self._flavors.extend(flavors)
        self._nsamp = samples
        
        (self._dist_dets, self._dist_samples) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp)

        self.data = {}
        for det in self._dist_dets:
            self.data[det] = {}
            for flv in self._flavors:
                self.data[det][flv] = np.zeros(self._dist_samples[1], dtype=np.float64)

        self.flags = {}
        for det in self._dist_dets:
            self.flags[det] = {}
            for flv in self._flavors:
                self.flags[det][flv] = np.zeros(self._dist_samples[1], dtype=np.uint8)


    @property
    def detectors(self):
        return self._dets

    @property
    def flavors(self):
        return self._flavors

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


    def _get(self, detector, flavor, start, n):
        return (self.data[detector][flavor][start:start+n], self.flags[detector][flavor][start:start+n])


    def _put(self, detector, flavor, start, data, flags):
        n = data.shape[0]
        self.data[detector][flavor][start:start+n] = np.copy(data)
        self.flags[detector][flavor][start:start+n] = np.copy(flags)
        return


    def read(self, detector=None, flavor='default', local_start=0, n=0):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if flavor not in self.flavors:
            raise ValueError('flavor {} not found'.format(flavor))
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError('local sample range {} - {} is invalid'.format(local_start, local_start+n-1))
        return self._get(detector, flavor, local_start, n)


    def write(self, detector=None, flavor='default', local_start=0, data=None, flags=None):
        if detector not in self.local_dets:
            raise ValueError('detector {} not found'.format(detector))
        if flavor not in self.flavors:
            raise ValueError('flavor {} not found'.format(flavor))
        if (data is None) or (flags is None):
            raise ValueError('both data and flags must be specified')
        if data.shape != flags.shape:
            raise ValueError('data and flags arrays must be the same length')
        if (local_start < 0) or (local_start + data.shape[0] > self.local_samples[1]):
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
        # would allocate memory for the data buffer of the base class.
        super().__init__(mpicomm=mpicomm, timedist=timedist, detectors=None, flavors=None, samples=0)

        if detectors is None:
            raise ValueError('you must specify a list of detectors')

        self._dets = detectors
        self._flavors = [self.DEFAULT_FLAVOR]
        self._nsamp = samples
        
        (self._dist_dets, self._dist_samples) = distribute_det_samples(self._mpicomm, self._timedist, self._dets, self._nsamp)

        self.rngstream = rngstream
        self.seeds = {}
        for det in enumerate(self._dets):
            self.seeds[det[1]] = det[0] 
        self.rms = rms


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

