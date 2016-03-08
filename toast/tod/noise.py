# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


from mpi4py import MPI

import unittest

import numpy as np


class Noise(object):
    """
    Base class for an object that describes the noise properties of all
    detectors for a single observation.

    Args:
        detectors (list): names of detectors we have
        freq (ndarray): array of frequencies for our PSDs
        psds (dict): dictionary of arrays which contains the PSD values
            for each detector.
    """

    def __init__(self, detectors=None, freq=np.ndarray([0.0]), psds=None):
        self._dets = []
        self._freq = freq
        self._nfreq = freq.shape[0]
        self._psds = {}
        if detectors is not None:
            self._dets = detectors
            if psds is None:
                raise RutimeError("you must specify a psd for each detector")
            for det in self._dets:
                if psds[det].shape[0] != self._nfreq:
                    raise RuntimeError("PSD length must match the number of frequencies")
                self._psds[det] = np.copy(psds[det])


    @property
    def detectors(self):
        return self._dets


    def multiply_ntt(self, detector, data):
        pass


    def multiply_invntt(self, detector, data):
        pass


    @property
    def freq(self):
        return self._freq
    

    def psd(self, detector):
        if detector not in self._dets:
            raise RuntimeError("psd for detector {} not found".format(detector))
        return self._psds[detector]

