# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import unittest

import numpy as np


class Noise(object):
    """
    Base class for an object that describes the noise properties of all
    detectors for a single observation.

    Args:
        detectors (list): names of detectors we have
        freqs (dict): dictionary of array of frequencies for our PSDs
        psds (dict): dictionary of arrays which contains the PSD values
            for each detector.
    """

    def __init__(self, detectors=None, freqs=None, psds=None):
        self._dets = []
        self._freqs = {}
        self._psds = {}
        self._weights = {}
        self._rate = {}

        if detectors is not None:
            self._dets = detectors
            if psds is None:
                raise RutimeError("you must provide a dictionary of PSD arrays for all detectors")
            if freqs is None:
                raise RutimeError("you must provide a dictionary of frequency arrays for all detectors")
            for det in self._dets:
                if det not in psds:
                    raise RuntimeError("no PSD specified for detector {}".format(det))
                if det not in freqs:
                    raise RuntimeError("no frequency array specified for detector {}".format(det))
                if psds[det].shape[0] != freqs[det].shape[0]:
                    raise RuntimeError("PSD length must match the number of frequencies")
                self._freqs[det] = np.copy(freqs[det])
                self._psds[det] = np.copy(psds[det])
                # last frequency point should be Nyquist
                self._rate[det] = 2.0 * self._freqs[det][-1]


    @property
    def detectors(self):
        """
        (list): list of strings containing the detector names.
        """
        return self._dets


    def multiply_ntt(self, detector, data):
        pass


    def multiply_invntt(self, detector, data):
        pass


    def freq(self, detector):
        """
        Get the frequencies for a detector.

        Args:
            detector (str): the detector name.

        Returns:
            (array): the frequency bins that are used for the PSD.
        """
        return self._freqs[detector]


    def rate(self, detector):
        """
        Get the sample rate for a detector.

        Args:
            detector (str): the detector name.

        Returns:
            (float): the sample rate in Hz.
        """
        return self._rate[detector]


    def psd(self, detector):
        """
        Get the PSD for a detector.

        Args:
            detector (str): the detector name.

        Returns:
            (array): an array containing the PSD for the detector.
        """
        if detector not in self._dets:
            raise RuntimeError("psd for detector {} not found".format(detector))
        return self._psds[detector]

