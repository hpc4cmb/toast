# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np


class Streams(object):
    """
    Base class for an object that provides a collection of streams.

    Each Streams class has one or more detectors, and each detctor
    might have different flavors of data and flags.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, flavors=None, samples=0):
        """
        Construct a Streams object given an MPI communicator.

        Args:
            mpicomm: the MPI communicator over which the data is distributed.
            timedist: if True, the data is distributed by time, otherwise by
                      detector.
            detectors: list of names to use for the detectors.
            flavors: list of *EXTRA* flavors to use (beyond the default).
            samples: pre-initialize the storage with this number of samples.

        Returns:
            Nothing

        Raises:
            Nothing
        """
        self.mpicomm = mpicomm
        self.timedist = timedist
        self.detectors = []
        if detectors != None:
            self.detectors = detectors
        self.flavors = ['default']
        if flavors != None:
            self.flavors.extend(flavors)
        self.samples = samples
        self.data = {}
        for det in detectors:
            self.data[det] = {}
            for flv in flavors:
                self.data[det][flv] = np.zeros(nsamp, dtype=np.float64)


    def _valid_dets(self):
        return self.detectors


    def _valid_flavors(self):
        return self.flavors


    def _nsamp(self):
        return self.samples


    def _get(self, detector, flavor, start, n):
        return self.data[detector][flavor]


    def _put(self, detector, flavor, start, data):
        self.data[detector][flavor][start:] = np.copy(data)


    def mpicomm(self):
        return self.mpicomm


    def read(self, detector=None, flavor='default', start=0, n=0):

        if detector not in _valid_dets():
            raise ValueError('detector {} not found'.format(detector))

        if flavor not in _valid_flavors():
            raise ValueError('flavor {} not found'.format(flavor))

        if (start < 0) or (start + n > _nsamp()):
            raise ValueError('sample range {} - {} is invalid'.format(start, start+n-1))

        return _get(self, detector, flavor, start, n)


    def write(self, detector=None, flavor='default', start=0, data=data):

        if detector not in _valid_dets():
            raise ValueError('detector {} not found'.format(detector))

        if flavor not in _valid_flavors():
            raise ValueError('flavor {} not found'.format(flavor))

        if (start < 0) or (start + data.shape[0] > _nsamp()):
            raise ValueError('sample range {} - {} is invalid'.format(start, start+data.shape[0]-1))

        _put(self, detector, flavor, start, data)



class StreamsWhiteNoise(Streams):
    """
    Provide white noise streams.
    """

    def __init__(self, mpicomm=MPI.COMM_WORLD, timedist=True, detectors=None, rms=1.0, samples=0, seed=0):
        """
        Construct a StreamsWhiteNoise object given an MPI communicator.

        Args:
            mpicomm: the MPI communicator over which the data is distributed.
            timedist: if True, the data is distributed by time, otherwise by
                      detector.
            detectors: list of names to use for the detectors.
            rms: RMS of the white noise.
            samples: pre-initialize the storage with this number of samples.

        Returns:
            Nothing

        Raises:
            Nothing
        """
        
        # We call the parent class constructor to set the MPI communicator and
        # distribution type, but we do NOT pass the detector list, as this 
        # would allocate memory for the data buffer in the base class.
        super().__init__(mpicomm=mpicomm, timdist=timedist, detectors=None, flavors=None, samples=0)

        if detectors is None:
            raise ValueError('you must specify a list of detectors')
        self.detectors = detectors
        self.seeds = {}
        for dets in enumerate(self.detectors):
            self.seeds[dets[1]] = dets[0] 
        self.flavors = ['default']
        self.samples = samples
        self.rms = rms


    def _valid_dets(self):
        return self.detectors


    def _valid_flavors(self):
        return self.flavors


    def _nsamp(self):
        return self.samples


    def _get(self, detector, flavor, start, n):
        # Setting the seed like this does NOT guarantee uncorrelated
        # results from the generator.  This is just a place holder until
        # the streamed rng is implemented.
        np.random.seed(self.seeds[detector])
        trash = np.random.norma(loc=0.0, scale=self.rms, size=(n-start))
        return np.random.normal(loc=0.0, scale=self.rms, size=n)


    def _put(self, detector, flavor, start, data):
        raise RuntimeError('cannot write data to simulated white noise streams')

